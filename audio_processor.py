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
    # ---- B轨数据源：主源 + 备用源自动 fallback ----
    # 主源（分析管线的默认数据源），停摆超过 FALLBACK_AFTER_SECONDS 自动切到备用源，
    # 主源恢复后自动切回。备用源的 processed/ 目录相互独立，互不污染。
    PRIMARY_SOURCE = os.getenv("B_TRACK_PRIMARY_SOURCE", "/Volumes/download/records/Sony-2")
    FALLBACK_SOURCES = [p.strip() for p in os.getenv(
        "B_TRACK_FALLBACK_SOURCES", "/Volumes/download/records/Sony-1").split(",") if p.strip()]
    FALLBACK_AFTER_SECONDS = int(os.getenv("B_TRACK_FALLBACK_AFTER", "600"))
    SOURCE_DIR = PRIMARY_SOURCE  # 兼容旧引用（= 主源常量）
    PROCESSED_DIR = "processed"
    FAILED_DIR = "failed"
    SCAN_INTERVAL = 3
    SUPPORTED_FORMATS = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']
    ASR_TRANSCRIBE_URL = os.getenv("ASR_TRANSCRIBE_URL", "http://localhost:5008/transcribes")

    # ---- Termux 上传停滞检测配置 ----
    STALL_DETECT_ENABLED = os.getenv("STALL_DETECT_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    # 白天（6:00~23:00）超过此秒数没有新文件到达则告警（默认 30 分钟）
    STALL_TIMEOUT_SECONDS = int(os.getenv("STALL_TIMEOUT_SECONDS", "1800"))
    # 两次告警之间的最小间隔（默认 2 小时），避免重复轰炸
    STALL_ALERT_COOLDOWN = int(os.getenv("STALL_ALERT_COOLDOWN", "7200"))
    # 告警/恢复前的复测观察窗口（秒）：窗口内新文件到达则判定为短暂停滞，静默跳过
    STALL_CONFIRM_WINDOW = int(os.getenv("STALL_CONFIRM_WINDOW", "180"))
    STALL_CONFIRM_INTERVAL = int(os.getenv("STALL_CONFIRM_INTERVAL", "20"))
    # 检测的活跃时段（小时），仅在此范围内检测
    STALL_ACTIVE_HOUR_START = 6
    STALL_ACTIVE_HOUR_END = 23

    # ---- Termux 停滞自动恢复配置 ----
    STALL_AUTO_RECOVERY_ENABLED = os.getenv("STALL_AUTO_RECOVERY_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    TERMUX_SSH_HOST = os.getenv("TERMUX_SSH_HOST", "192.168.1.193")
    TERMUX_SSH_PORT = int(os.getenv("TERMUX_SSH_PORT", "8022"))
    TERMUX_SSH_USER = os.getenv("TERMUX_SSH_USER", "root")
    TERMUX_SSH_PASSWORD = os.getenv("TERMUX_SSH_PASSWORD", "")
    TERMUX_RECOVERY_COMMAND = os.getenv(
        "TERMUX_RECOVERY_COMMAND",
        "/data/data/com.termux/files/home/all_in_one.sh start"
    )
    # 自动恢复最小间隔（默认 30 分钟），避免停滞期间每分钟重复重启
    STALL_RECOVERY_COOLDOWN = int(os.getenv("STALL_RECOVERY_COOLDOWN", "1800"))
    STALL_RECOVERY_TIMEOUT = int(os.getenv("STALL_RECOVERY_TIMEOUT", "90"))


def _admin_headers():
    """调用 5008 的写操作/转写接口时附带管理令牌（.env 的 ADMIN_TOKEN，未配置则不附带）"""
    token = (os.getenv("ADMIN_TOKEN") or "").strip()
    return {"X-Admin-Token": token} if token else {}


# ---- 上传活跃度全局状态 ----
_last_new_file_time = time.time()          # 最后一次发现新文件的时间戳
_last_stall_alert_time = 0.0               # 上次发送停滞告警的时间戳
_last_stall_recovery_time = 0.0            # 上次自动恢复的时间戳
_last_mount_alert_time = 0.0               # 上次发送挂载异常告警的时间戳
_stall_active_since = 0.0                  # 当前停滞事件开始时间；0 表示未处于停滞事件
_stall_status_lock = threading.Lock()

# ---- B轨数据源 fallback 状态 ----
_active_source_dir = FileMonitorConfig.PRIMARY_SOURCE
_source_switch_lock = threading.Lock()
_fallback_cutoff_time = 0.0   # fallback 生效时刻-10分钟缓冲；fallback 源只处理晚于该时间的录音，
                              # 更早的积压是主源活着时段的重复录音（时间轴已由主源覆盖），永不处理


def get_active_source():
    """B轨当前的扫描/处理数据源（主源或 fallback 源）"""
    return _active_source_dir


def _dir_readable_quick(path, timeout=15.0):
    """带超时探测目录是否可读（网络目录可能挂起，必须子线程+超时）"""
    result = {"ok": False}

    def _probe():
        try:
            os.listdir(path)
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout)
    return result["ok"]


def _apply_source_change(new_dir, reason):
    """执行数据源切换：更新状态、重建 processed/failed 目录、重置停滞基准时间"""
    global _active_source_dir, _last_new_file_time, _stall_active_since, _fallback_cutoff_time
    if new_dir != FileMonitorConfig.PRIMARY_SOURCE:
        # 只处理"主源真实缺席时段"的录音：切换前 10 分钟缓冲（主源停摆判定阈值），
        # 更早的 Sony-1 积压与主源已处理时间轴重复，跳过
        _fallback_cutoff_time = time.time() - FileMonitorConfig.FALLBACK_AFTER_SECONDS
    else:
        _fallback_cutoff_time = 0.0
    with _source_switch_lock:
        _active_source_dir = new_dir
    # 重置停滞基准：以新源的最新文件时间为准（含 processed/，待处理文件会被秒级移走）
    new_mtime = max(
        _get_latest_file_mtime(new_dir),
        _get_latest_file_mtime(new_dir, include_processed=True),
    )
    with _stall_status_lock:
        _last_new_file_time = new_mtime if new_mtime > 0 else time.time()
        _stall_active_since = 0.0
    logger.warning(f"🔀 B轨数据源已切换: {new_dir}（{reason}），"
                   f"停滞基准重置为 {datetime.fromtimestamp(_last_new_file_time).strftime('%H:%M:%S')}")


def check_and_switch_source():
    """看门狗每轮调用：主源停摆超阈值→切备用源；主源恢复→切回。
    切换与恢复均发送通知邮件。
    注意：主源存活判定用"待处理文件 + processed/ 最新文件"的 max——
    待处理文件会被秒级移走，只扫源目录会把正常状态误判成停摆。"""
    now = time.time()
    primary_mtime = max(
        _get_latest_file_mtime(FileMonitorConfig.PRIMARY_SOURCE),
        _get_latest_file_mtime(FileMonitorConfig.PRIMARY_SOURCE, include_processed=True),
    )
    primary_alive = (primary_mtime > 0 and (now - primary_mtime) <= FileMonitorConfig.FALLBACK_AFTER_SECONDS)

    with _source_switch_lock:
        current = _active_source_dir

    if current == FileMonitorConfig.PRIMARY_SOURCE:
        if primary_alive or not FileMonitorConfig.FALLBACK_SOURCES:
            return
        for cand in FileMonitorConfig.FALLBACK_SOURCES:
            if _dir_readable_quick(cand):
                _apply_source_change(cand, f"主源停摆超过 {FileMonitorConfig.FALLBACK_AFTER_SECONDS // 60} 分钟")
                try:
                    from email_utils import send_email_sync
                    send_email_sync(
                        "🔀 B轨已自动切换到备用录音源",
                        f"主源 {FileMonitorConfig.PRIMARY_SOURCE} 停摆超过 "
                        f"{FileMonitorConfig.FALLBACK_AFTER_SECONDS // 60} 分钟无新文件。\n\n"
                        f"B轨分析已自动切换到备用源: {cand}\n"
                        f"客厅的哭声分析不会中断；主源恢复后将自动切回。"
                    )
                except Exception as e:
                    logger.error(f"切换通知邮件发送失败: {e}")
                return
        logger.error(f"⛔ 主源停摆但备用源均不可读: {FileMonitorConfig.FALLBACK_SOURCES}")
    else:
        if primary_alive:
            old = current
            _apply_source_change(FileMonitorConfig.PRIMARY_SOURCE, "主源已恢复")
            try:
                from email_utils import send_email_sync
                send_email_sync(
                    "✅ B轨已切回主录音源",
                    f"主源 {FileMonitorConfig.PRIMARY_SOURCE} 已恢复新文件，"
                    f"B轨分析已从备用源 {old} 切回主源。"
                )
            except Exception as e:
                logger.error(f"切回通知邮件发送失败: {e}")


def _probe_nas_mount(timeout=15.0):
    """探测 SMB 挂载是否可读（带超时）。
    NAS 挂载假死时读取会无限挂起，必须子线程+超时。"""
    result = {"ok": False}

    def _probe():
        try:
            os.listdir(get_active_source())
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout)
    return result["ok"]


def update_last_file_time():
    """由监控循环在发现新文件时调用，更新活跃时间戳"""
    global _last_new_file_time
    with _stall_status_lock:
        _last_new_file_time = time.time()

def _get_latest_file_mtime(source_dir=None, include_processed=False):
    """快速扫描最近文件夹获取最新文件时间（默认扫描当前 B 轨数据源）。
    include_processed=True 时同时考虑 processed/ 里的文件——待处理文件会被
    处理线程秒级移走，仅扫源目录会误判为"无活动"；processed/ 里的文件
    才是稳定的"最近真实上传"信号（fallback 判定必须用它）。"""
    latest_mtime = 0
    try:
        if source_dir is None:
            source_dir = get_active_source()
        if not os.path.exists(source_dir):
            return 0

        today = datetime.now()
        yesterday = today - timedelta(days=1)
        recent_folders = [today.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")]

        if include_processed:
            pdir = os.path.join(source_dir, FileMonitorConfig.PROCESSED_DIR)
            for d in recent_folders:
                sub = os.path.join(pdir, d)
                try:
                    for subitem in _safe_listdir(sub):
                        if subitem.startswith('.'):
                            continue
                        subp = os.path.join(sub, subitem)
                        if os.path.isfile(subp) and os.path.splitext(subitem)[1].lower() in FileMonitorConfig.SUPPORTED_FORMATS:
                            mtime = os.path.getmtime(subp)
                            if mtime > latest_mtime:
                                latest_mtime = mtime
                except Exception:
                    pass
        
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
                    for subitem in _safe_listdir(item_path):
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


def _safe_get_latest_file_mtime(timeout=20.0):
    """带超时地扫描最新文件时间。网络目录扫描可能挂起，
    在子线程中执行，超时后返回 0 并跳过本次，避免看门狗线程被永久阻塞。"""
    result = {"mtime": 0}

    def _scan():
        try:
            result["mtime"] = _get_latest_file_mtime()
        except Exception as e:
            logger.error(f"安全扫描最新文件时间异常: {e}")

    t = threading.Thread(target=_scan, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.warning(f"⚠️ 扫描最新文件时间超时({timeout}s)，本次跳过")
        return 0
    return result["mtime"]


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


def _attempt_termux_recovery(elapsed_min, last_time_str):
    """通过 SSH 执行 Termux 录音上传恢复命令。"""
    if not FileMonitorConfig.STALL_AUTO_RECOVERY_ENABLED:
        return {
            "attempted": False,
            "ok": False,
            "message": "自动恢复未启用",
            "output": "",
        }

    ssh_bin = shutil.which("ssh")
    if not ssh_bin:
        return {
            "attempted": False,
            "ok": False,
            "message": "本机缺少 ssh 命令",
            "output": "",
        }

    cmd = [
        ssh_bin,
        "-o", "PubkeyAuthentication=no",
        "-o", "PreferredAuthentications=password,keyboard-interactive",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/tmp/asr_termux_recovery_known_hosts",
        "-o", "ConnectTimeout=10",
        "-p", str(FileMonitorConfig.TERMUX_SSH_PORT),
        f"{FileMonitorConfig.TERMUX_SSH_USER}@{FileMonitorConfig.TERMUX_SSH_HOST}",
        FileMonitorConfig.TERMUX_RECOVERY_COMMAND,
    ]

    sshpass_bin = shutil.which("sshpass")
    env = None
    if FileMonitorConfig.TERMUX_SSH_PASSWORD:
        if not sshpass_bin:
            return {
                "attempted": False,
                "ok": False,
                "message": "已配置 SSH 密码，但本机缺少 sshpass",
                "output": "",
            }
        env = os.environ.copy()
        env["SSHPASS"] = FileMonitorConfig.TERMUX_SSH_PASSWORD
        cmd = [sshpass_bin, "-e"] + cmd

    target = f"{FileMonitorConfig.TERMUX_SSH_USER}@{FileMonitorConfig.TERMUX_SSH_HOST}:{FileMonitorConfig.TERMUX_SSH_PORT}"
    logger.warning(
        f"🛠️ 尝试自动恢复 Termux 上传: target={target}, "
        f"停滞={elapsed_min}分钟, 上次文件={last_time_str}"
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FileMonitorConfig.STALL_RECOVERY_TIMEOUT,
            env=env,
        )
        output = _join_process_output(result.stdout, result.stderr)
        ok = result.returncode == 0
        if ok:
            logger.info(f"✅ Termux 自动恢复命令执行成功\n{output}")
            message = "自动恢复命令执行成功"
        else:
            logger.error(f"❌ Termux 自动恢复命令执行失败，返回码={result.returncode}\n{output}")
            message = f"自动恢复命令执行失败，返回码={result.returncode}"
        return {
            "attempted": True,
            "ok": ok,
            "message": message,
            "output": output,
        }
    except subprocess.TimeoutExpired as e:
        output = _join_process_output(e.stdout, e.stderr)
        logger.error(f"❌ Termux 自动恢复命令超时 ({FileMonitorConfig.STALL_RECOVERY_TIMEOUT}s)\n{output}")
        return {
            "attempted": True,
            "ok": False,
            "message": f"自动恢复命令超时 ({FileMonitorConfig.STALL_RECOVERY_TIMEOUT}s)",
            "output": output,
        }
    except Exception as e:
        logger.error(f"❌ Termux 自动恢复异常: {e}")
        return {
            "attempted": True,
            "ok": False,
            "message": f"自动恢复异常: {e}",
            "output": "",
        }


def _send_stall_recovered_email(stall_active_since, last_file_time):
    """停滞事件结束后发送恢复通知。"""
    try:
        from email_utils import send_email_sync

        now = datetime.now()
        stalled_minutes = max(0, int((last_file_time - stall_active_since) // 60))
        stall_start_str = datetime.fromtimestamp(stall_active_since).strftime("%Y-%m-%d %H:%M:%S")
        last_file_str = datetime.fromtimestamp(last_file_time).strftime("%Y-%m-%d %H:%M:%S")

        subject = "✅ Termux 录音上传已恢复"
        content = (
            f"✅ Termux 录音上传已恢复\n"
            f"{'=' * 40}\n\n"
            f"恢复确认时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"停滞开始时间: {stall_start_str}\n"
            f"最新收到文件时间: {last_file_str}\n"
            f"本次停滞约: {stalled_minutes} 分钟\n"
            f"监控目录: {FileMonitorConfig.SOURCE_DIR}\n\n"
            f"系统已经重新检测到新音频文件到达，Termux 上传链路恢复正常。\n"
            f"{'=' * 40}"
        )
        send_email_sync(subject, content)
        logger.info("📧 Termux 上传恢复邮件已发送")
    except Exception as e:
        logger.error(f"❌ 发送 Termux 上传恢复邮件失败: {e}")


def _stall_watchdog():
    """
    后台看门狗线程：定期检查 SOURCE_DIR 是否长时间没有新音频文件到达。
    若超过阈值且处于活跃时段，发送邮件告警。
    """
    global _last_new_file_time, _last_stall_alert_time, _last_stall_recovery_time, _stall_active_since
    global _last_mount_alert_time

    logger.info("🐕 Termux 上传停滞看门狗已启动")
    logger.info(f"   停滞阈值: {FileMonitorConfig.STALL_TIMEOUT_SECONDS}秒, "
                f"告警冷却: {FileMonitorConfig.STALL_ALERT_COOLDOWN}秒, "
                f"自动恢复: {FileMonitorConfig.STALL_AUTO_RECOVERY_ENABLED}, "
                f"恢复冷却: {FileMonitorConfig.STALL_RECOVERY_COOLDOWN}秒, "
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

            # ---- B轨数据源自动 fallback：主源停摆→切备用源；主源恢复→切回 ----
            # （在停滞判定之前执行；切换会重置停滞基准，本轮按新源判断）
            check_and_switch_source()

            # 独立扫描获取真实最新的文件时间，避免受处理线程阻塞的影响
            actual_latest = _safe_get_latest_file_mtime()
            if actual_latest > 0:
                with _stall_status_lock:
                    if actual_latest > _last_new_file_time:
                        _last_new_file_time = actual_latest

            with _stall_status_lock:
                elapsed = time.time() - _last_new_file_time
                last_file_time = _last_new_file_time

            if elapsed <= FileMonitorConfig.STALL_TIMEOUT_SECONDS:
                stall_to_notify = 0.0
                with _stall_status_lock:
                    if _stall_active_since > 0:
                        stall_to_notify = _stall_active_since
                        _stall_active_since = 0.0

                if stall_to_notify > 0:
                    _send_stall_recovered_email(stall_to_notify, last_file_time)

                continue  # 正常，还在阈值内

            # ---- 触发停滞处理 ----
            elapsed_min = int(elapsed // 60)
            last_time_str = datetime.fromtimestamp(last_file_time).strftime("%Y-%m-%d %H:%M:%S")

            # ---- 挂载健康检查：挂载不通 ≠ 手机上传停止 ----
            if not _probe_nas_mount():
                logger.warning("📡 NAS 挂载不可读（手机上传可能正常，是 Mac 侧挂载假死），"
                               "不触发上传停滞告警和手机重启，等待挂载看门狗修复")
                if (time.time() - _last_mount_alert_time) >= FileMonitorConfig.STALL_ALERT_COOLDOWN:
                    _last_mount_alert_time = time.time()
                    try:
                        from email_utils import send_email_sync
                        send_email_sync(
                            "NAS 挂载异常（非手机上传问题）",
                            f"Mac 侧 SMB 挂载不可读，已 {elapsed_min} 分钟看不到新文件。\n"
                            "挂载看门狗会自动重挂（通常 1-3 分钟内恢复），手机上传不受影响。\n"
                            f"上次文件时间: {last_time_str}"
                        )
                    except Exception as e:
                        logger.error(f"挂载异常邮件发送失败: {e}")
                continue

            # 冷却短路：恢复与告警都在冷却期内时，仅记录日志等待下一轮
            recovery_due = (time.time() - _last_stall_recovery_time) >= FileMonitorConfig.STALL_RECOVERY_COOLDOWN
            alert_due = (time.time() - _last_stall_alert_time) >= FileMonitorConfig.STALL_ALERT_COOLDOWN
            if not (recovery_due or alert_due):
                logger.warning(f"🚨 Termux 上传停滞持续 {elapsed_min} 分钟 "
                               f"(上次文件: {last_time_str})，冷却等待中")
                continue

            logger.warning(f"🚨 Termux 上传停滞 {elapsed_min} 分钟 "
                           f"(上次文件: {last_time_str})，开始复测确认...")

            # ---- 复测：发告警/执行恢复前观察一小段时间，
            # 排除"正在自行恢复（新文件迟到）"的抖动，避免发出滞后告警 ----
            recheck_mtime = 0
            confirm_deadline = time.time() + FileMonitorConfig.STALL_CONFIRM_WINDOW
            while time.time() < confirm_deadline:
                time.sleep(FileMonitorConfig.STALL_CONFIRM_INTERVAL)
                recheck_mtime = _safe_get_latest_file_mtime()
                if recheck_mtime > last_file_time:
                    break

            if recheck_mtime > last_file_time:
                with _stall_status_lock:
                    if recheck_mtime > _last_new_file_time:
                        _last_new_file_time = recheck_mtime
                logger.info(f"👀 复测发现新文件已到达 "
                            f"({datetime.fromtimestamp(recheck_mtime).strftime('%H:%M:%S')})，"
                            f"判定为短暂停滞，跳过本次告警与恢复")
                continue

            # ---- 复测确认仍无新文件：真停滞 ----
            with _stall_status_lock:
                if _stall_active_since <= 0:
                    _stall_active_since = last_file_time

            recovery_result = {
                "attempted": False,
                "ok": False,
                "message": "自动恢复冷却中，未重复执行",
                "output": "",
            }
            if recovery_due:
                recovery_result = _attempt_termux_recovery(elapsed_min, last_time_str)
                _last_stall_recovery_time = time.time()

            # 检查邮件告警冷却期。自动恢复和邮件告警分开冷却：
            # 恢复可以按较短间隔重试，邮件仍避免重复轰炸。
            if alert_due:
                try:
                    from email_utils import send_email_sync
                    recovery_state = "已尝试自动恢复" if recovery_result["attempted"] else "未执行自动恢复"
                    if recovery_result["attempted"] and recovery_result["ok"]:
                        recovery_state = "自动恢复命令成功"
                    elif recovery_result["attempted"]:
                        recovery_state = "自动恢复命令失败"

                    subject = f"⚠️ Termux 录音上传停滞告警 - 已停止 {elapsed_min} 分钟（{recovery_state}）"
                    content = (
                        f"⚠️  Termux 录音上传停滞告警\n"
                        f"{'=' * 40}\n\n"
                        f"检测时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"上次收到文件: {last_time_str}\n"
                        f"已停滞时长: {elapsed_min} 分钟\n"
                        f"监控目录: {FileMonitorConfig.SOURCE_DIR}\n\n"
                        f"{'=' * 40}\n"
                        f"自动恢复:\n"
                        f"  状态: {recovery_result['message']}\n"
                        f"  目标: {FileMonitorConfig.TERMUX_SSH_USER}@{FileMonitorConfig.TERMUX_SSH_HOST}:{FileMonitorConfig.TERMUX_SSH_PORT}\n"
                        f"  命令: {FileMonitorConfig.TERMUX_RECOVERY_COMMAND}\n"
                        f"  恢复冷却期: {FileMonitorConfig.STALL_RECOVERY_COOLDOWN // 60} 分钟\n\n"
                        f"恢复命令输出:\n"
                        f"{recovery_result['output'] or '(无输出)'}\n\n"
                        f"{'=' * 40}\n"
                        f"可能原因:\n"
                        f"  1. Termux 应用崩溃或被系统杀死\n"
                        f"  2. 手机录音服务停止\n"
                        f"  3. 网络/NAS 挂载异常\n\n"
                        f"系统已按配置尝试自动恢复；若后续仍无新文件，请检查手机 Termux 状态。\n"
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
    # 网络目录（SMB/NFS）扫描可能挂起，用线程+超时保护，避免阻塞启动
    _scan_thread = threading.Thread(target=_init_last_file_time, daemon=True)
    _scan_thread.start()
    _scan_thread.join(10)
    if _scan_thread.is_alive():
        logger.warning("⚠️ 看门狗基准时间扫描超时(10s)，已跳过，改用当前时间作为基准")

    # 启动上传停滞看门狗
    if FileMonitorConfig.STALL_DETECT_ENABLED:
        watchdog_thread = threading.Thread(target=_stall_watchdog, daemon=True)
        watchdog_thread.start()

    thread = threading.Thread(target=_monitor_loop, daemon=True)
    thread.start()
    return thread


_HANGING_DIRS = set()  # 已确认会挂起的网络目录，避免反复探测


def _safe_listdir(path, timeout=3.0):
    """安全列出目录。网络目录（SMB/NFS）的 readdir 可能永久挂起，
    用子线程 + 超时保护，避免阻塞主流程；挂起的目录会被记录并跳过。"""
    normalized = os.path.normpath(path)
    if normalized in _HANGING_DIRS:
        return []
    result = []

    def _do():
        try:
            result.extend(os.listdir(path))
        except Exception as e:
            logger.warning(f"⚠️ 读取目录异常 {path}: {e}")

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.warning(f"⚠️ 读取目录超时({timeout}s)，疑似挂起，本次跳过: {path}")
        _HANGING_DIRS.add(normalized)
        return []
    return result


def _init_last_file_time():
    """扫描当前数据源获取最新文件的修改时间，作为看门狗基准"""
    global _last_new_file_time
    try:
        source_dir = get_active_source()
        latest_mtime = 0
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
            for item in sorted(_safe_listdir(FileMonitorConfig.SOURCE_DIR)):
                item_path = os.path.join(FileMonitorConfig.SOURCE_DIR, item)
                
                if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR, 
                            "audio_segments", "logs"] or item.startswith('.'):
                    continue
                
                # 只处理日期格式的子目录 (YYYY-MM-DD)
                date_str = _extract_date_from_filename(item)
                if not date_str or not os.path.isdir(item_path):
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
    current_source = get_active_source()
    # 统一归档：无论当前源是 Sony-2 还是 fallback 的 Sony-1，处理结果一律归档到
    # 主源 processed/（按录音时间分日期）——时间轴永远完整连续，历史补跑只看一个地方
    processed_dir = os.path.join(FileMonitorConfig.PRIMARY_SOURCE, FileMonitorConfig.PROCESSED_DIR)
    failed_dir = os.path.join(FileMonitorConfig.PRIMARY_SOURCE, FileMonitorConfig.FAILED_DIR)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    while True:
        try:
            # 数据源可能被 fallback 机制切换（Sony-2 停摆时切到 Sony-1，恢复后切回）
            source_dir = get_active_source()
            if source_dir != current_source:
                current_source = source_dir
                logger.info(f"📂 B轨实时监听目录: {current_source}"
                            f"（处理结果统一归档至 {FileMonitorConfig.PRIMARY_SOURCE}/{FileMonitorConfig.PROCESSED_DIR}）")

            if not os.path.exists(source_dir):
                logger.warning(f"⚠️ 源目录不存在: {source_dir}")
                time.sleep(FileMonitorConfig.SCAN_INTERVAL)
                continue

            # fallback 源的旧积压是"主源活着时段的重复录音"，日期目录级直接跳过（零开销）
            cutoff_date = None
            if current_source != FileMonitorConfig.PRIMARY_SOURCE and _fallback_cutoff_time > 0:
                cutoff_date = datetime.fromtimestamp(_fallback_cutoff_time).strftime("%Y-%m-%d")

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
                    # 日期目录（YYYY-MM-DD）：早于 cutoff 的整个目录跳过，不做 listdir
                    if (cutoff_date and len(item) == 10 and item[4] == '-' and item[7] == '-'
                            and item < cutoff_date):
                        continue
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
                        filename not in processed_files):
                        # 文件级 cutoff：fallback 当天的旧文件（cutoff 之前的时段主源活着，已覆盖）
                        if cutoff_date:
                            rt = parse_recording_time(filename)
                            if rt is None or rt.timestamp() < _fallback_cutoff_time:
                                continue
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

        # 【2026-09-21】历史掉队文件防线：录音时间超过 6 小时的文件不进实时管线。
        # 手机端积压队列补传会让几天前的录音此刻才落到 NAS（如 Sony-1 在 09-09
        # 傍晚停滞期的积压），若照常送 ASR，A 轨会把它们逐个当成"实时哭声"触发
        # 即时报警+Webhook，造成凌晨轰炸式"重复告警"。
        # 历史文件的哭声检测由补跑脚本（reprocess_history_cries.py）统一负责，
        # 那条链路不会触发报警。文件仍归档到 processed/，未打 DB 标记，
        # 补跑跑到对应月份时会正常补检。
        _age_sec = (datetime.now() - recording_time).total_seconds()
        if _age_sec > 6 * 3600:
            logger.info(f"⏭️ 跳过历史掉队文件 (录音于 {_age_sec/3600:.1f} 小时前): {filename}")
            _move_file(filepath, filename, processed_dir, recording_time)
            return True

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
                                     data={'source_device': source_device},
                                     headers=_admin_headers(), timeout=7200)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ 转录完成: {filename} ({len(result.get('full_text', ''))} 字)")
            _move_file(filepath, filename, processed_dir, recording_time)
            return True
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
