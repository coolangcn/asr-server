#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import logging
import threading
import shutil
import requests
import traceback
from datetime import datetime, timedelta
from db_manager import parse_recording_time, is_file_processed_a, mark_file_processed_a

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger('AudioProcessor')

class FileMonitorConfig:
    ENABLED = True
    SOURCE_DIR = "/Volumes/download/records/Sony-2"
    PROCESSED_DIR = "processed"
    FAILED_DIR = "failed"
    SCAN_INTERVAL = 3
    SUPPORTED_FORMATS = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']
    ASR_TRANSCRIBE_URL = os.getenv("ASR_TRANSCRIBE_URL", "http://localhost:5008/transcribes")

    # ---- Termux 上传停滞检测配置 ----
    STALL_DETECT_ENABLED = True
    # 白天（6:00~23:00）超过此秒数没有新文件到达则告警（默认 30 分钟）
    STALL_TIMEOUT_SECONDS = int(os.getenv("STALL_TIMEOUT_SECONDS", "1800"))
    # 两次告警之间的最小间隔（默认 2 小时），避免重复轰炸
    STALL_ALERT_COOLDOWN = int(os.getenv("STALL_ALERT_COOLDOWN", "7200"))
    # 检测的活跃时段（小时），仅在此范围内检测
    STALL_ACTIVE_HOUR_START = 6
    STALL_ACTIVE_HOUR_END = 23


# ---- 上传活跃度全局状态 ----
_last_new_file_time = time.time()          # 最后一次发现新文件的时间戳
_last_stall_alert_time = 0.0               # 上次发送停滞告警的时间戳
_stall_status_lock = threading.Lock()


def update_last_file_time():
    """由监控循环在发现新文件时调用，更新活跃时间戳"""
    global _last_new_file_time
    with _stall_status_lock:
        _last_new_file_time = time.time()

def _get_latest_file_mtime():
    """快速扫描最近文件夹获取最新文件时间"""
    latest_mtime = 0
    try:
        source_dir = FileMonitorConfig.SOURCE_DIR
        if not os.path.exists(source_dir):
            return 0
            
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        recent_folders = [today.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")]
        
        for item in os.listdir(source_dir):
            if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR,
                        "audio_segments", "logs"] or item.startswith('.'):
                continue
            item_path = os.path.join(source_dir, item)
            
            items_to_check = []
            if os.path.isfile(item_path):
                items_to_check.append(item_path)
            elif os.path.isdir(item_path) and item in recent_folders:
                try:
                    for subitem in os.listdir(item_path):
                        if not subitem.startswith('.'):
                            subp = os.path.join(item_path, subitem)
                            if os.path.isfile(subp):
                                items_to_check.append(subp)
                except Exception:
                    pass
            
            for filepath in items_to_check:
                ext = os.path.splitext(filepath)[1].lower()
                if ext in FileMonitorConfig.SUPPORTED_FORMATS:
                    mtime = os.path.getmtime(filepath)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
    except Exception as e:
        logger.error(f"获取最新文件时间失败: {e}")
    return latest_mtime

def get_stall_status():
    """返回当前停滞检测状态（供 API 层调用）"""
    with _stall_status_lock:
        elapsed = time.time() - _last_new_file_time
        is_stalled = elapsed > FileMonitorConfig.STALL_TIMEOUT_SECONDS
        return {
            "last_file_time": datetime.fromtimestamp(_last_new_file_time).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed),
            "is_stalled": is_stalled,
            "threshold_seconds": FileMonitorConfig.STALL_TIMEOUT_SECONDS,
        }


def _stall_watchdog():
    """
    后台看门狗线程：定期检查 SOURCE_DIR 是否长时间没有新音频文件到达。
    若超过阈值且处于活跃时段，发送邮件告警。
    """
    global _last_new_file_time, _last_stall_alert_time

    logger.info("🐕 Termux 上传停滞看门狗已启动")
    logger.info(f"   停滞阈值: {FileMonitorConfig.STALL_TIMEOUT_SECONDS}秒, "
                f"告警冷却: {FileMonitorConfig.STALL_ALERT_COOLDOWN}秒, "
                f"活跃时段: {FileMonitorConfig.STALL_ACTIVE_HOUR_START}:00 ~ "
                f"{FileMonitorConfig.STALL_ACTIVE_HOUR_END}:00")

    # 检测间隔：每 60 秒检查一次
    CHECK_INTERVAL = 60

    while True:
        try:
            time.sleep(CHECK_INTERVAL)

            now = datetime.now()
            current_hour = now.hour

            # 仅在活跃时段检测
            if not (FileMonitorConfig.STALL_ACTIVE_HOUR_START <= current_hour < FileMonitorConfig.STALL_ACTIVE_HOUR_END):
                continue

            # 独立扫描获取真实最新的文件时间，避免受处理线程阻塞的影响
            actual_latest = _get_latest_file_mtime()
            if actual_latest > 0:
                with _stall_status_lock:
                    if actual_latest > _last_new_file_time:
                        _last_new_file_time = actual_latest

            with _stall_status_lock:
                elapsed = time.time() - _last_new_file_time

            if elapsed <= FileMonitorConfig.STALL_TIMEOUT_SECONDS:
                continue  # 正常，还在阈值内

            # 检查冷却期
            if (time.time() - _last_stall_alert_time) < FileMonitorConfig.STALL_ALERT_COOLDOWN:
                continue  # 冷却中，不重复发送

            # ---- 触发告警 ----
            elapsed_min = int(elapsed // 60)
            last_time_str = datetime.fromtimestamp(_last_new_file_time).strftime("%Y-%m-%d %H:%M:%S")

            logger.warning(f"🚨 Termux 上传停滞告警！已 {elapsed_min} 分钟没有新文件到达 "
                           f"(上次文件: {last_time_str})")

            try:
                from email_utils import send_email_sync
                subject = f"⚠️ Termux 录音上传停滞告警 - 已停止 {elapsed_min} 分钟"
                content = (
                    f"⚠️  Termux 录音上传停滞告警\n"
                    f"{'=' * 40}\n\n"
                    f"检测时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"上次收到文件: {last_time_str}\n"
                    f"已停滞时长: {elapsed_min} 分钟\n"
                    f"监控目录: {FileMonitorConfig.SOURCE_DIR}\n\n"
                    f"{'=' * 40}\n"
                    f"可能原因:\n"
                    f"  1. Termux 应用崩溃或被系统杀死\n"
                    f"  2. 手机录音服务停止\n"
                    f"  3. 网络/NAS 挂载异常\n\n"
                    f"请检查手机 Termux 状态并恢复录音上传。\n"
                    f"{'=' * 40}\n"
                    f"此告警冷却期: {FileMonitorConfig.STALL_ALERT_COOLDOWN // 60} 分钟\n"
                    f"（同一问题不会在冷却期内重复发送）"
                )
                send_email_sync(subject, content)
                _last_stall_alert_time = time.time()
                logger.info("📧 停滞告警邮件已发送")
            except Exception as e:
                logger.error(f"❌ 发送停滞告警邮件失败: {e}")

        except Exception as e:
            logger.error(f"看门狗线程异常: {e}")


def start_monitor():
    """启动文件监控并自动处理新音频"""
    if not FileMonitorConfig.ENABLED:
        logger.info("📂 文件监控功能已禁用")
        return

    # 初始化看门狗基准时间：取 SOURCE_DIR 中最新文件的修改时间
    _init_last_file_time()

    # 启动上传停滞看门狗
    if FileMonitorConfig.STALL_DETECT_ENABLED:
        watchdog_thread = threading.Thread(target=_stall_watchdog, daemon=True)
        watchdog_thread.start()

    thread = threading.Thread(target=_monitor_loop, daemon=True)
    thread.start()
    return thread


def _init_last_file_time():
    """扫描 SOURCE_DIR 获取最新文件的修改时间，作为看门狗基准"""
    global _last_new_file_time
    try:
        source_dir = FileMonitorConfig.SOURCE_DIR
        latest_mtime = 0
        for item in os.listdir(source_dir):
            if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR,
                        "audio_segments", "logs"] or item.startswith('.'):
                continue
            item_path = os.path.join(source_dir, item)
            
            items_to_check = []
            if os.path.isfile(item_path):
                items_to_check.append(item_path)
            elif os.path.isdir(item_path):
                try:
                    for subitem in os.listdir(item_path):
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
            logger.info(f"🐕 看门狗基准时间已初始化: "
                        f"{datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.info("🐕 看门狗基准时间: 当前时间（SOURCE_DIR 中无文件）")
    except Exception as e:
        logger.warning(f"⚠️ 初始化看门狗基准时间失败: {e}")

def _extract_date_from_filename(filename):
    """从文件名或路径中提取日期 (YYYY-MM-DD)"""
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None

def _monitor_loop():
    logger.info("📂 B轨文件监控已启动")
    logger.info(f"   监控目录: {FileMonitorConfig.SOURCE_DIR}")
    
    # 确保必要的目录存在
    processed_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, FileMonitorConfig.PROCESSED_DIR)
    os.makedirs(processed_dir, exist_ok=True)
    failed_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, FileMonitorConfig.FAILED_DIR)
    os.makedirs(failed_dir, exist_ok=True)
    
    # ==================== 阶段一：Catch-up 全量历史追赶 ====================
    logger.info("🚀 【阶段一：Catch-up】开始全量扫描历史文件...")
    date_files = {}  # {date_str: [(filename, filepath), ...]}
    total_scanned = 0
    total_skipped = 0
    
    if os.path.exists(FileMonitorConfig.SOURCE_DIR):
        try:
            for item in sorted(os.listdir(FileMonitorConfig.SOURCE_DIR)):
                item_path = os.path.join(FileMonitorConfig.SOURCE_DIR, item)
                
                if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR, 
                            "audio_segments", "logs"] or item.startswith('.'):
                    continue
                
                # 只处理日期格式的子目录 (YYYY-MM-DD)
                date_str = _extract_date_from_filename(item)
                if not date_str or not os.path.isdir(item_path):
                    continue
                
                date_files.setdefault(date_str, [])
                
                for subitem in os.listdir(item_path):
                    if subitem.startswith('.'):
                        continue
                    filepath = os.path.join(item_path, subitem)
                    if not os.path.isfile(filepath):
                        continue
                    
                    ext = os.path.splitext(subitem)[1].lower()
                    if ext not in FileMonitorConfig.SUPPORTED_FORMATS or 'TEMP' in subitem:
                        continue
                    
                    total_scanned += 1
                    
                    # 跳过 A 轨已处理的文件
                    if is_file_processed_a(subitem):
                        total_skipped += 1
                        continue
                    
                    date_files[date_str].append((subitem, filepath))
        except Exception as e:
            logger.error(f"❌ Catch-up 扫描异常: {e}")
            logger.error(traceback.format_exc())
    
    # 按日期排序（从老到新）
    sorted_dates = sorted(date_files.keys())
    total_catchup = sum(len(files) for files in date_files.values())
    
    logger.info(f"📊 【Catch-up 扫描完成】共 {total_scanned} 个历史文件，"
                f"跳过 A 轨已处理 {total_skipped} 个，"
                f"待处理 {total_catchup} 个，跨越 {len(sorted_dates)} 天")
    
    # 逐日期处理
    catchup_processed = 0
    catchup_failed = 0
    for date_idx, date_str in enumerate(sorted_dates, 1):
        files = date_files[date_str]
        if not files:
            continue
        
        logger.info(f"📅 [{date_idx}/{len(sorted_dates)}] 处理日期 {date_str}，共 {len(files)} 个文件")
        
        # 按文件名排序
        files.sort(key=lambda x: x[0])
        
        for file_idx, (filename, filepath) in enumerate(files, 1):
            try:
                # 双重检查：处理前再次确认未被 A 轨处理
                if is_file_processed_a(filename):
                    logger.info(f"  ⏭️ [{file_idx}/{len(files)}] {filename} — A轨已处理，跳过")
                    continue
                
                recording_time = parse_recording_time(filename)
                if recording_time:
                    hour = recording_time.hour
                    if 1 <= hour < 6:
                        logger.info(f"  ⏭️ [{file_idx}/{len(files)}] {filename} — 凌晨录音，跳过")
                        _move_file(filepath, filename, processed_dir, recording_time)
                        mark_file_processed_a(filename, status="skipped_night")
                        catchup_processed += 1
                        continue
                
                logger.info(f"  📤 [{file_idx}/{len(files)}] {filename} — 开始处理")
                success = _process_one_file_b(filename, filepath, processed_dir, failed_dir)
                if success:
                    mark_file_processed_a(filename, status="b_catchup_success")
                catchup_processed += 1
            except Exception as e:
                logger.error(f"  ❌ [{file_idx}/{len(files)}] {filename} — 处理失败: {e}")
                catchup_failed += 1
        
        logger.info(f"  ✅ {date_str} 完成，成功 {catchup_processed}，失败 {catchup_failed}")
    
    logger.info(f"🎉 【阶段一：Catch-up】全量历史追赶完成！"
                f"共处理 {catchup_processed} 个文件，失败 {catchup_failed} 个")
    logger.info("🔄 【阶段二：Real-time】切换到实时监听模式，等待新文件到达...")
    
    # ==================== 阶段二：Real-time 实时监听 ====================
    processed_files = set()  # 内存缓存，避免重复处理
    
    while True:
        try:
            if not os.path.exists(FileMonitorConfig.SOURCE_DIR):
                logger.warning(f"⚠️ 源目录不存在: {FileMonitorConfig.SOURCE_DIR}")
                time.sleep(FileMonitorConfig.SCAN_INTERVAL)
                continue
            
            files_to_process = []
            for item in os.listdir(FileMonitorConfig.SOURCE_DIR):
                item_path = os.path.join(FileMonitorConfig.SOURCE_DIR, item)
                
                if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR, 
                            "audio_segments", "logs"] or item.startswith('.'):
                    continue
                
                items_to_check = []
                if os.path.isfile(item_path):
                    items_to_check.append(item_path)
                elif os.path.isdir(item_path):
                    try:
                        for subitem in os.listdir(item_path):
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
                        filename not in processed_files):
                        # 实时模式下也检查 A 轨进度
                        if not is_file_processed_a(filename):
                            files_to_process.append((filename, filepath))
                        else:
                            processed_files.add(filename)  # 加入内存缓存，避免重复检查
            
            # 按文件名排序
            files_to_process.sort(key=lambda x: x[0])
            
            if files_to_process:
                update_last_file_time()  # 刷新看门狗时间戳
                logger.info(f"🔍 [Real-time] 发现 {len(files_to_process)} 个新文件")
                for filename, filepath in files_to_process:
                    try:
                        recording_time = parse_recording_time(filename)
                        success = _process_one_file_b(filename, filepath, processed_dir, failed_dir)
                        if success:
                            mark_file_processed_a(filename, status="b_realtime_success")
                        processed_files.add(filename)
                    except Exception as e:
                        logger.error(f"处理文件 {filename} 失败: {e}")
            
        except Exception as e:
            logger.error(f"监控循环异常: {e}")
            logger.error(traceback.format_exc())
            
        time.sleep(FileMonitorConfig.SCAN_INTERVAL)

def _process_one_file_b(filename, filepath, processed_dir, failed_dir):
    """处理单个音频文件（B轨），返回是否成功"""
    # 1. 检查录音时间，跳过凌晨 1-6 点
    recording_time = parse_recording_time(filename)
    if recording_time:
        hour = recording_time.hour
        if 1 <= hour < 6:
            logger.info(f"⏭️ 跳过凌晨录音: {filename}")
            _move_file(filepath, filename, processed_dir, recording_time)
            return True

    logger.info(f"📤 开始处理: {filename}")
    
    # 2. 发起转录请求
    try:
        with open(filepath, 'rb') as f:
            files_data = {'audio_file': (filename, f, 'audio/mpeg')}
            response = requests.post(FileMonitorConfig.ASR_TRANSCRIBE_URL, files=files_data, timeout=7200)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ 转录完成: {filename} ({len(result.get('full_text', ''))} 字)")
            _move_file(filepath, filename, processed_dir, recording_time)
            return True
        elif response.status_code == 503:
            logger.info(f"⏳ B 轨分析当前由于 A 轨任务而暂停，文件保留在原处等待重试: {filename}")
            return False  # 不移动文件，等待下一轮扫描
        else:
            logger.error(f"❌ 转录失败: {filename} (HTTP {response.status_code})")
            _move_file(filepath, filename, failed_dir, recording_time)
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
