#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import logging
import threading
import shutil
import subprocess
import requests
import traceback
from datetime import datetime, timedelta
from db_manager import parse_recording_time, is_file_processed_a, mark_file_processed_a

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

logger = logging.getLogger('AudioProcessor')

class FileMonitorConfig:
    ENABLED = True
    # ---- B轨数据源：双 Pixel 双主源（2026-10 起全面转向 Pixel 录音）----
    # 所有源地位平等：同时扫描、同时转录（A 轨哭声 + B 轨声纹），各自 processed/
    # 按设备独立归档，互不污染。
    SOURCES = [p.strip() for p in os.getenv(
        "B_TRACK_SOURCES",
        "/Volumes/download/records/Pixel-6,/Volumes/download/records/Pixel-5"
    ).split(",") if p.strip()]
    SOURCE_DIR = SOURCES[0]  # 兼容旧引用（asr_server 等处的默认源 alias）
    PROCESSED_DIR = "processed"
    FAILED_DIR = "failed"
    SCAN_INTERVAL = 3
    SUPPORTED_FORMATS = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']
    ASR_TRANSCRIBE_URL = os.getenv("ASR_TRANSCRIBE_URL", "http://localhost:5008/transcribes")


def _admin_headers():
    """调用 5008 的写操作/转写接口时附带管理令牌（.env 的 ADMIN_TOKEN，未配置则不附带）"""
    token = (os.getenv("ADMIN_TOKEN") or "").strip()
    return {"X-Admin-Token": token} if token else {}


# ---- 上传活跃度状态（供 get_stall_status 仪表盘展示）----
_last_new_file_time = time.time()          # 最后一次发现新文件的时间戳
_stall_status_lock = threading.Lock()


def update_last_file_time():
    """由监控循环在发现新文件时调用，更新活跃时间戳"""
    global _last_new_file_time
    with _stall_status_lock:
        _last_new_file_time = time.time()


def get_stall_status():
    """返回扫描活跃度状态（供 API 层仪表盘调用）。
    【2026-10-03 双源改造】fallback/停滞看门狗已随单源架构移除，
    仅保留"最近发现新文件"的活跃度信息；is_stalled 恒为 False。"""
    with _stall_status_lock:
        elapsed = time.time() - _last_new_file_time
        return {
            "last_file_time": datetime.fromtimestamp(_last_new_file_time).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed),
            "is_stalled": False,
            "threshold_seconds": 0,
        }


def _truncate_text(text, limit=3000):
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _join_process_output(*parts):
    lines = []
    for part in parts:
        if not part:
            continue
        lines.append(_truncate_text(part, limit=100000))
    return _truncate_text("\n".join(lines))


def start_monitor():
    """启动文件监控并自动处理新音频"""
    if not FileMonitorConfig.ENABLED:
        logger.info("📂 文件监控功能已禁用")
        return

    # 初始化活跃度基准时间：取各数据源中最新文件的修改时间
    # 网络目录（SMB/NFS）扫描可能挂起，用线程+超时保护，避免阻塞启动
    _scan_thread = threading.Thread(target=_init_last_file_time, daemon=True)
    _scan_thread.start()
    _scan_thread.join(10)
    if _scan_thread.is_alive():
        logger.warning("⚠️ 活跃度基准时间扫描超时(10s)，已跳过，改用当前时间作为基准")

    thread = threading.Thread(target=_monitor_loop, daemon=True)
    thread.start()
    return thread


_HANGING_DIRS = {}  # {目录: 允许重试的时间戳}——超时目录 5 分钟后允许重试，防止永久拉黑
_HANGING_RETRY_SEC = 300


def _hanging_check(normalized):
    """目录仍在拉黑期返回 True；过期则解除拉黑返回 False"""
    expire = _HANGING_DIRS.get(normalized)
    if expire is None:
        return False
    if time.time() >= expire:
        del _HANGING_DIRS[normalized]
        logger.info(f"🔄 拉黑目录重试窗口开启: {normalized}")
        return False
    return True


_SMB_SCAN_SEM = threading.Semaphore(2)  # 全局 SMB 扫描限流：防止启动期多线程并发 listdir 拥塞内核 smbfs 队列

# B 轨监听心跳：smb_watchdog 检测该文件的 mtime，停滞超阈值即判定监听线程
# 内核 SMB 调用悬死（假死），自动执行"杀进程→重挂→拉起"自愈（2026-10-03）
_B_TRACK_HEARTBEAT = os.getenv("B_TRACK_HEARTBEAT_FILE", "/Users/mac/asr-server/log/b_track_heartbeat")
_smb_fail_since = None  # 连续 SMB 超时的起始时间；任一次成功即清零（None）


def _touch_heartbeat():
    """刷新监听心跳（mtime）。文件不存在则创建，存在则仅更新时间戳。
    【2026-10-03】连续 SMB 超时超过 180s 时刻意停滞心跳——循环活着但 SMB 全悬死
    同样是假死，让 smb_watchdog 接管自愈。"""
    try:
        if _smb_fail_since is not None and (time.time() - _smb_fail_since) > 180:
            return
        if os.path.exists(_B_TRACK_HEARTBEAT):
            os.utime(_B_TRACK_HEARTBEAT)
        else:
            with open(_B_TRACK_HEARTBEAT, "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass  # 心跳失败绝不影响监听本身


def _safe_listdir(path, timeout=3.0):
    """安全列出目录。网络目录（SMB/NFS）的 readdir 可能永久挂起，
    用子线程 + 超时保护，避免阻塞主流程；挂起的目录会被记录并跳过。"""
    normalized = os.path.normpath(path)
    if _hanging_check(normalized):
        return []
    result = []

    def _do():
        try:
            result.extend(os.listdir(path))
        except FileNotFoundError:
            pass  # 目录尚不存在（如当日无归档文件）视为空目录，不刷告警
        except Exception as e:
            logger.warning(f"⚠️ 读取目录异常 {path}: {e}")

    with _SMB_SCAN_SEM:
        t = threading.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout)
    if t.is_alive():
        logger.warning(f"⚠️ 读取目录超时({timeout}s)，疑似挂起，本次跳过: {path}")
        _HANGING_DIRS[normalized] = time.time() + _HANGING_RETRY_SEC
        global _smb_fail_since
        if _smb_fail_since is None:
            _smb_fail_since = time.time()  # 连续失败计时起点
        return []
    if result:
        # 本次成功 → 清零连续失败计时
        _smb_fail_since = None
    return result


def _init_last_file_time():
    """扫描全部数据源获取最新文件的修改时间，作为活跃度基准"""
    global _last_new_file_time
    try:
        latest_mtime = 0
        for source_dir in FileMonitorConfig.SOURCES:
            for item in _safe_listdir(source_dir):
                if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR,
                            "audio_segments", "logs"] or item.startswith('.'):
                    continue
                item_path = os.path.join(source_dir, item)

                items_to_check = []
                if os.path.isfile(item_path):
                    items_to_check.append(item_path)
                elif os.path.isdir(item_path):
                    try:
                        for subitem in _safe_listdir(item_path):
                            if not subitem.startswith('.'):
                                subp = os.path.join(item_path, subitem)
                                if os.path.isfile(subp):
                                    items_to_check.append(subp)
                    except Exception as e:
                        logger.error(f"读取子目录 {item_path} 失败: {e}")

                for filepath in items_to_check:
                    filename = os.path.basename(filepath)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in FileMonitorConfig.SUPPORTED_FORMATS:
                        mtime = os.path.getmtime(filepath)
                        latest_mtime = max(latest_mtime, mtime)

        if latest_mtime > 0:
            with _stall_status_lock:
                _last_new_file_time = latest_mtime
            logger.info(f"📊 活跃度基准时间已初始化: "
                        f"{datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.info("📊 活跃度基准时间: 当前时间（各数据源中无文件）")
    except Exception as e:
        logger.warning(f"⚠️ 初始化活跃度基准时间失败: {e}")

def _extract_date_from_filename(filename):
    """从文件名或路径中提取日期 (YYYY-MM-DD)"""
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None

def _monitor_loop():
    logger.info("📂 B轨文件监控已启动（多源并行）")
    for _s in FileMonitorConfig.SOURCES:
        logger.info(f"   监控目录: {_s}")

    # 各源独立建 processed/failed 目录（按设备归档，互不污染）
    for source_dir in FileMonitorConfig.SOURCES:
        os.makedirs(os.path.join(source_dir, FileMonitorConfig.PROCESSED_DIR), exist_ok=True)
        os.makedirs(os.path.join(source_dir, FileMonitorConfig.FAILED_DIR), exist_ok=True)

    # ==================== 阶段一：Catch-up 全量历史追赶 ====================
    logger.info("🚀 【阶段一：Catch-up】开始扫描最近7天历史文件（更早的归补跑脚本负责）...")
    _catchup_min_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    catchup_processed = 0
    catchup_failed = 0

    for source_dir in FileMonitorConfig.SOURCES:
        device = os.path.basename(source_dir.rstrip("/"))
        processed_dir = os.path.join(source_dir, FileMonitorConfig.PROCESSED_DIR)
        failed_dir = os.path.join(source_dir, FileMonitorConfig.FAILED_DIR)

        date_files = {}  # {date_str: [(filename, filepath), ...]}
        total_scanned = 0
        total_skipped = 0

        if os.path.exists(source_dir):
            try:
                for item in sorted(_safe_listdir(source_dir)):
                    item_path = os.path.join(source_dir, item)

                    if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR,
                                "audio_segments", "logs"] or item.startswith('.'):
                        continue

                    # 只处理日期格式的子目录 (YYYY-MM-DD)
                    date_str = _extract_date_from_filename(item)
                    if not date_str or not os.path.isdir(item_path):
                        continue

                    # 【2026-10-03】只追最近7天：历史目录在 SMB 上扫描极慢且易超时，
                    # 几千个老目录会把启动拖死；老积压由 reprocess_history_cries.py 补跑负责
                    if date_str < _catchup_min_date:
                        continue

                    date_files.setdefault(date_str, [])

                    for subitem in _safe_listdir(item_path):
                        if subitem.startswith('.'):
                            continue
                        filepath = os.path.join(item_path, subitem)
                        if not os.path.isfile(filepath):
                            continue

                        ext = os.path.splitext(subitem)[1].lower()
                        if ext not in FileMonitorConfig.SUPPORTED_FORMATS or 'TEMP' in subitem:
                            continue

                        total_scanned += 1

                        # 跳过 A 轨已处理的文件（按设备标记，兼容旧裸文件名标记）
                        if is_file_processed_a(subitem, device):
                            total_skipped += 1
                            continue

                        date_files[date_str].append((subitem, filepath))
            except Exception as e:
                logger.error(f"❌ Catch-up 扫描异常 [{device}]: {e}")
                logger.error(traceback.format_exc())
        # 按日期排序（从老到新）
        sorted_dates = sorted(date_files.keys())
        total_catchup = sum(len(files) for files in date_files.values())

        logger.info(f"📊 【Catch-up 扫描完成·{device}】共 {total_scanned} 个历史文件，"
                    f"跳过 A 轨已处理 {total_skipped} 个，"
                    f"待处理 {total_catchup} 个，跨越 {len(sorted_dates)} 天")

        # 逐日期处理
        for date_idx, date_str in enumerate(sorted_dates, 1):
            files = date_files[date_str]
            if not files:
                continue

            logger.info(f"📅 [{device}] [{date_idx}/{len(sorted_dates)}] 处理日期 {date_str}，共 {len(files)} 个文件")

            # 按文件名排序
            files.sort(key=lambda x: x[0])

            for file_idx, (filename, filepath) in enumerate(files, 1):
                _touch_heartbeat()  # 【2026-10-03】每个文件处理前刷新心跳，catch-up 长循环不假死
                try:
                    # 双重检查：处理前再次确认未被 A 轨处理
                    if is_file_processed_a(filename, device):
                        logger.info(f"  ⏭️ [{file_idx}/{len(files)}] {filename} — A轨已处理，跳过")
                        continue

                    recording_time = parse_recording_time(filename)
                    if recording_time:
                        hour = recording_time.hour
                        if 1 <= hour < 6:
                            logger.info(f"  ⏭️ [{file_idx}/{len(files)}] {filename} — 凌晨录音，跳过")
                            _move_file(filepath, filename, processed_dir, recording_time)
                            mark_file_processed_a(filename, status="skipped_night", device=device)
                            catchup_processed += 1
                            continue

                    logger.info(f"  📤 [{file_idx}/{len(files)}] {filename} — 开始处理")
                    result = _process_one_file_b(filename, filepath, processed_dir, failed_dir)
                    if result:
                        mark_file_processed_a(filename, status=_b_result_status(result, "b_catchup_success"),
                                              device=device)
                    catchup_processed += 1
                except Exception as e:
                    logger.error(f"  ❌ [{file_idx}/{len(files)}] {filename} — 处理失败: {e}")
                    catchup_failed += 1

            logger.info(f"  ✅ {date_str} 完成（{device}），累计成功 {catchup_processed}，失败 {catchup_failed}")

    logger.info(f"🎉 【阶段一：Catch-up】全量历史追赶完成！"
                f"共处理 {catchup_processed} 个文件，失败 {catchup_failed} 个")
    logger.info("🔄 【阶段二：Real-time】切换到实时监听模式，等待新文件到达...")
    
    # ==================== 阶段二：Real-time 实时监听 ====================
    # 内存缓存按设备隔离：两台 Pixel 同秒录音文件名相同，不能共用一个 set
    processed_files = {s: set() for s in FileMonitorConfig.SOURCES}

    while True:
        _touch_heartbeat()  # 【2026-10-03】每轮刷新监听心跳（smb_watchdog 检测假死）
        try:
            for source_dir in FileMonitorConfig.SOURCES:
                device = os.path.basename(source_dir.rstrip("/"))
                processed_dir = os.path.join(source_dir, FileMonitorConfig.PROCESSED_DIR)
                failed_dir = os.path.join(source_dir, FileMonitorConfig.FAILED_DIR)
                known = processed_files.setdefault(source_dir, set())

                if not os.path.exists(source_dir):
                    logger.warning(f"⚠️ 源目录不存在: {source_dir}")
                    continue

                files_to_process = []
                for item in _safe_listdir(source_dir):
                    item_path = os.path.join(source_dir, item)

                    if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR,
                                "audio_segments", "logs"] or item.startswith('.'):
                        continue

                    items_to_check = []
                    if os.path.isfile(item_path):
                        items_to_check.append(item_path)
                    elif os.path.isdir(item_path):
                        try:
                            for subitem in _safe_listdir(item_path):
                                if not subitem.startswith('.'):
                                    subp = os.path.join(item_path, subitem)
                                    if os.path.isfile(subp):
                                        items_to_check.append(subp)
                        except Exception as e:
                            logger.error(f"读取子目录 {item_path} 失败: {e}")

                    for filepath in items_to_check:
                        filename = os.path.basename(filepath)
                        ext = os.path.splitext(filename)[1].lower()

                        if (ext in FileMonitorConfig.SUPPORTED_FORMATS and
                            'TEMP' not in filename and
                            filename not in known):
                            # 实时模式下也检查 A 轨进度（按设备标记，兼容旧裸文件名标记）
                            if not is_file_processed_a(filename, device):
                                files_to_process.append((filename, filepath))
                            else:
                                known.add(filename)  # 加入内存缓存，避免重复检查

                # 按文件名排序
                files_to_process.sort(key=lambda x: x[0])

                if files_to_process:
                    update_last_file_time()  # 刷新活跃度时间戳
                    logger.info(f"🔍 [Real-time·{device}] 发现 {len(files_to_process)} 个新文件")
                    for filename, filepath in files_to_process:
                        try:
                            result = _process_one_file_b(filename, filepath, processed_dir, failed_dir)
                            if result:
                                mark_file_processed_a(filename, status=_b_result_status(result, "b_realtime_success"),
                                                      device=device)
                                known.add(filename)  # 只有成功才入缓存；失败（网络异常等）下一轮重扫自动重试
                        except Exception as e:
                            logger.error(f"处理文件 {filename} 失败: {e}")

        except Exception as e:
            logger.error(f"监控循环异常: {e}")
            logger.error(traceback.format_exc())

        time.sleep(FileMonitorConfig.SCAN_INTERVAL)

# 【2026-10-04】B 轨处理结果 → A 轨标记状态映射。
# dropped（掉队防线，>6h 归档但从未送检）才是真·待补救；no_speech/night_skip 表示
# 文件已按预期送检处理完（只是无声或夜间降级未转写），绝不能计入待补救——否则每天
# 新增的静音录音会让 web_viewer 的「待补救」数单调增长且永远降不到 0（用户反馈）。
_B_RESULT_STATUS = {
    "no_speech": "b_no_speech",
    "night_skip": "b_night_skip",
    "dropped": "b_dropped_history",
}


def _b_result_status(result, base_success):
    """把 _process_one_file_b 的返回值翻译成 A 轨标记状态。
    base_success 为 'b_catchup_success'（catch-up）或 'b_realtime_success'（实时）。"""
    return _B_RESULT_STATUS.get(result, base_success)


def _process_one_file_b(filename, filepath, processed_dir, failed_dir):
    """处理单个音频文件（B轨）。
    返回 str 表示已处理完（'success' 有语音入库 / 'no_speech' 无声 / 'night_skip' 夜间降级 /
    'dropped' 历史掉队归档未送检）；返回 False 表示失败，文件保留原处等下轮重试。"""
    # 1. 检查录音时间
    recording_time = parse_recording_time(filename)
    _skip_asr = False
    if recording_time:
        hour = recording_time.hour
        _age_sec = (datetime.now() - recording_time).total_seconds()
        # 【2026-10-04 夜间降级】凌晨 1-6 点录音：跳过 ASR 转写（GPU 让给凌晨哭声补跑），
        # 但保留轨道A哭声检测+即时告警——夜间哭声值守不再盲区（原逻辑是完全跳过不送检）。
        # 仅对 6 小时内的新鲜录音生效；更旧的走下方历史掉队防线（交给补跑链路，防止轰炸告警）。
        if 1 <= hour < 6 and _age_sec <= 6 * 3600:
            _skip_asr = True
            logger.info(f"🌙 夜间降级 (仅哭声检测): {filename}")

        # 【2026-09-21】历史掉队文件防线：录音时间超过 6 小时的文件不进实时管线。
        # 手机端积压队列补传会让几天前的录音此刻才落到 NAS（如 Sony-1 在 09-09
        # 傍晚停滞期的积压），若照常送 ASR，A 轨会把它们逐个当成"实时哭声"触发
        # 即时报警+Webhook，造成凌晨轰炸式"重复告警"。
        # 历史文件的哭声检测由补跑脚本（reprocess_history_cries.py）统一负责，
        # 那条链路不会触发报警。文件仍归档到 processed/，未打 DB 标记，
        # 补跑跑到对应月份时会正常补检。
        if _age_sec > 6 * 3600:
            logger.info(f"⏭️ 跳过历史掉队文件 (录音于 {_age_sec/3600:.1f} 小时前): {filename}")
            _move_file(filepath, filename, processed_dir, recording_time)
            return "dropped"  # 归档但从未送检 —— 唯一真·待补救来源，用 b_dropped_history 显式标记

    logger.info(f"📤 开始处理: {filename}")
    
    # 2. 发起转录请求
    try:
        # 从 NAS 路径推导来源设备名（.../records/Sony-2/2026-09-20/xxx.m4a → Sony-2），
        # 供 5008 哭声报警 Webhook 标注报警来源
        _m_dev = re.search(r"records/([^/]+)/", filepath.replace("\\", "/"))
        source_device = _m_dev.group(1) if _m_dev else ""
        with open(filepath, 'rb') as f:
            files_data = {'audio_file': (filename, f, 'audio/mpeg')}
            response = requests.post(FileMonitorConfig.ASR_TRANSCRIBE_URL, files=files_data,
                                     data={'source_device': source_device,
                                           'skip_asr': 'true' if _skip_asr else 'false'},
                                     headers=_admin_headers(), timeout=7200)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ 转录完成: {filename} ({len(result.get('full_text', ''))} 字)")
            _move_file(filepath, filename, processed_dir, recording_time)
            if _skip_asr:
                return "night_skip"  # 夜间降级：已送检（哭声检测照跑），但未转写，不算待补救
            if not (result.get('segments') or []):
                return "no_speech"  # 已送检、VAD 0 段：分析已完成，只是无声，不算待补救
            return "success"
        elif response.status_code == 401:
            # 鉴权失败属配置问题，绝非文件本身有问题——绝不能把录音挪进 failed/ 隔离
            logger.error(
                f"🔑 转录鉴权失败(401): {filename} —— .env 的 ADMIN_TOKEN 与 5008 服务端不一致，"
                f"请检查配置后重启。文件保留在原处等待重试。"
            )
            return False  # 不移动文件，等待修复配置后重试
        elif response.status_code == 503:
            logger.info(f"⏳ B 轨分析当前由于 A 轨任务而暂停，文件保留在原处等待重试: {filename}")
            return False  # 不移动文件，等待下一轮扫描
        else:
            logger.error(f"❌ 转录失败: {filename} (HTTP {response.status_code})")
            _move_file(filepath, filename, failed_dir, recording_time)
            return False
            
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # 5008 重启窗口/网络抖动不是文件的问题——保留原处等下轮重试, 绝不能归档 failed/
        logger.warning(f"⏳ 5008 不可达（{type(e).__name__}），文件保留在原处等待重试: {filename}")
        return False
    except Exception as e:
        logger.error(f"❌ 处理文件 {filename} 时发生异常: {e}")
        _move_file(filepath, filename, failed_dir, recording_time)
        return False

def _move_file(src_path, filename, base_dest_dir, recording_time=None):
    """根据日期子目录移动文件"""
    date_subdir = recording_time.strftime("%Y-%m-%d") if recording_time else datetime.now().strftime("%Y-%m-%d")
    target_dir = os.path.join(base_dest_dir, date_subdir)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    
    try:
        shutil.move(src_path, target_path)
        logger.info(f"📦 已移动至: {os.path.basename(base_dest_dir)}/{date_subdir}/{filename}")
    except Exception as e:
        logger.warning(f"⚠️ 移动文件失败: {e}")
