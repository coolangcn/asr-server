#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import logging
import threading
import shutil
import subprocess
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

logger = logging.getLogger('RecoveryMonitor')


class DeviceConfig:
    def __init__(self, name, source_dir, ssh_host, ssh_port=8022, ssh_user="root",
                 ssh_password="", recovery_command="", enabled=True,
                 stall_timeout=1800, recovery_cooldown=1800, recovery_timeout=90,
                 active_hour_start=6, active_hour_end=23, room="unknown"):
        self.name = name
        self.source_dir = source_dir
        self.ssh_host = ssh_host
        self.ssh_port = int(ssh_port)
        self.ssh_user = ssh_user
        self.ssh_password = ssh_password
        self.recovery_command = recovery_command
        self.enabled = enabled
        self.stall_timeout = int(stall_timeout)
        self.recovery_cooldown = int(recovery_cooldown)
        self.recovery_timeout = int(recovery_timeout)
        self.active_hour_start = int(active_hour_start)
        self.active_hour_end = int(active_hour_end)
        self.room = room


ROOM_LABELS = {"living": "客厅", "bedroom": "卧室"}


class DeviceState:
    def __init__(self):
        self.last_new_file_time = time.time()
        self.last_stall_alert_time = 0.0
        self.last_stall_recovery_time = 0.0
        self.last_mount_alert_time = 0.0
        self.last_room_degrade_log_time = 0.0
        self.stall_active_since = 0.0
        self.lock = threading.Lock()


def _room_has_other_active_device(device):
    """房间视角：同房间其他设备是否仍在录音（有新文件）。用于降级单台故障的告警。"""
    now = time.time()
    for other in _devices:
        if other.name == device.name or not other.enabled or other.room != device.room:
            continue
        other_state = _device_states.get(other.name)
        if not other_state:
            continue
        with other_state.lock:
            if now - other_state.last_new_file_time <= other.stall_timeout:
                return True
    return False


def _parse_devices_from_env():
    devices = []
    device_names_str = os.getenv("RECOVERY_DEVICE_NAMES", "")
    if not device_names_str:
        return devices

    device_names = [n.strip() for n in device_names_str.split(",") if n.strip()]
    base_dir = os.getenv("RECOVERY_BASE_DIR", "/Volumes/download/records")

    for name in device_names:
        env_prefix = f"RECOVERY_{name.upper().replace('-', '_')}_"

        source_dir = os.getenv(
            f"{env_prefix}SOURCE_DIR",
            os.path.join(base_dir, name)
        )
        ssh_host = os.getenv(f"{env_prefix}SSH_HOST", "")
        if not ssh_host:
            logger.warning(f"⚠️ 设备 {name} 未配置 SSH_HOST，跳过")
            continue

        ssh_port = os.getenv(f"{env_prefix}SSH_PORT", "8022")
        ssh_user = os.getenv(f"{env_prefix}SSH_USER", "root")
        ssh_password = os.getenv(f"{env_prefix}SSH_PASSWORD", "")
        recovery_command = os.getenv(
            f"{env_prefix}RECOVERY_COMMAND",
            "/data/data/com.termux/files/home/all_in_one.sh start"
        )
        enabled = os.getenv(f"{env_prefix}ENABLED", "true").lower() in ("1", "true", "yes", "on")
        stall_timeout = os.getenv(f"{env_prefix}STALL_TIMEOUT", "1800")
        recovery_cooldown = os.getenv(f"{env_prefix}RECOVERY_COOLDOWN", "1800")
        recovery_timeout = os.getenv(f"{env_prefix}RECOVERY_TIMEOUT", "90")
        active_hour_start = os.getenv(f"{env_prefix}ACTIVE_HOUR_START", "6")
        active_hour_end = os.getenv(f"{env_prefix}ACTIVE_HOUR_END", "23")
        room = os.getenv(f"{env_prefix}ROOM", "unknown").strip().lower()

        device = DeviceConfig(
            name=name,
            source_dir=source_dir,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            ssh_password=ssh_password,
            recovery_command=recovery_command,
            enabled=enabled,
            stall_timeout=stall_timeout,
            recovery_cooldown=recovery_cooldown,
            recovery_timeout=recovery_timeout,
            active_hour_start=active_hour_start,
            active_hour_end=active_hour_end,
            room=room,
        )
        devices.append(device)
        logger.info(f"📱 已配置恢复监控设备: {name} ({ssh_host}) -> {source_dir} [房间: {room}]")

    return devices


_devices = []
_device_states = {}
_initialized = False


def _get_latest_file_mtime(device):
    latest_mtime = 0
    try:
        source_dir = device.source_dir
        if not os.path.exists(source_dir):
            return 0

        today = datetime.now()
        yesterday = today - timedelta(days=1)
        recent_folders = [today.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")]

        supported_formats = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']

        for item in os.listdir(source_dir):
            if item in ["processed", "failed", "audio_segments", "logs"] or item.startswith('.'):
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
                if ext in supported_formats:
                    mtime = os.path.getmtime(filepath)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
    except Exception as e:
        logger.error(f"[{device.name}] 获取最新文件时间失败: {e}")
    return latest_mtime


def _get_latest_file_mtime_with_timeout(device, timeout=45.0):
    """带超时地扫描最新文件时间。SMB 挂载假死时 listdir/stat 会无限挂起，
    必须在子线程中执行，超时后返回 0（按无新文件处理），避免看门狗线程被永久阻塞。"""
    result = {"mtime": 0}

    def _scan():
        result["mtime"] = _get_latest_file_mtime(device)

    t = threading.Thread(target=_scan, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.warning(f"[{device.name}] 目录扫描超时 ({timeout}s)，本轮按无新文件处理")
        return 0
    return result["mtime"]


def _probe_nas_mount(source_dir, timeout=15.0):
    """探测 SMB 挂载是否可读（带超时）。挂载不可读 ≠ 手机上传停止，
    此时应等挂载看门狗修复，而不是 SSH 重启手机。"""
    result = {"ok": False}

    def _probe():
        try:
            os.listdir(source_dir)
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout)
    return result["ok"]


def _attempt_ssh_recovery(device, elapsed_min, last_time_str):
    if not device.enabled:
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
        "-o", f"UserKnownHostsFile=/tmp/asr_recovery_{device.name}_known_hosts",
        "-o", "ConnectTimeout=10",
        "-p", str(device.ssh_port),
        f"{device.ssh_user}@{device.ssh_host}",
        device.recovery_command,
    ]

    sshpass_bin = shutil.which("sshpass")
    env = None
    if device.ssh_password:
        if not sshpass_bin:
            return {
                "attempted": False,
                "ok": False,
                "message": "已配置 SSH 密码，但本机缺少 sshpass",
                "output": "",
            }
        env = os.environ.copy()
        env["SSHPASS"] = device.ssh_password
        cmd = [sshpass_bin, "-e"] + cmd

    target = f"{device.ssh_user}@{device.ssh_host}:{device.ssh_port}"
    logger.warning(
        f"🛠️ [{device.name}] 尝试自动恢复上传: target={target}, "
        f"停滞={elapsed_min}分钟, 上次文件={last_time_str}"
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=device.recovery_timeout,
            env=env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output[-2000:] if len(output) > 2000 else output
        ok = result.returncode == 0
        if ok:
            logger.info(f"✅ [{device.name}] 自动恢复命令执行成功\n{output}")
            message = "自动恢复命令执行成功"
        else:
            logger.error(f"❌ [{device.name}] 自动恢复命令执行失败，返回码={result.returncode}\n{output}")
            message = f"自动恢复命令执行失败，返回码={result.returncode}"
        return {
            "attempted": True,
            "ok": ok,
            "message": message,
            "output": output,
        }
    except subprocess.TimeoutExpired as e:
        output = (getattr(e, 'stdout', '') or "") + (getattr(e, 'stderr', '') or "")
        output = output[-2000:] if len(output) > 2000 else output
        logger.error(f"❌ [{device.name}] 自动恢复命令超时 ({device.recovery_timeout}s)\n{output}")
        return {
            "attempted": True,
            "ok": False,
            "message": f"自动恢复命令超时 ({device.recovery_timeout}s)",
            "output": output,
        }
    except Exception as e:
        logger.error(f"❌ [{device.name}] 自动恢复异常: {e}")
        return {
            "attempted": True,
            "ok": False,
            "message": f"自动恢复异常: {e}",
            "output": "",
        }


def _send_stall_alert_email(device, elapsed_min, last_time_str, recovery_result):
    try:
        from email_utils import send_email_sync

        now = datetime.now()
        recovery_state = "已尝试自动恢复" if recovery_result["attempted"] else "未执行自动恢复"
        if recovery_result["attempted"] and recovery_result["ok"]:
            recovery_state = "自动恢复命令成功"
        elif recovery_result["attempted"]:
            recovery_state = "自动恢复命令失败"

        subject = f"⚠️ {device.name} 录音上传停滞告警 - 已停止 {elapsed_min} 分钟（{recovery_state}）"
        content = (
            f"⚠️  {device.name} 录音上传停滞告警\n"
            f"{'=' * 40}\n\n"
            f"检测时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"上次收到文件: {last_time_str}\n"
            f"已停滞时长: {elapsed_min} 分钟\n"
            f"监控目录: {device.source_dir}\n\n"
            f"{'=' * 40}\n"
            f"自动恢复:\n"
            f"  状态: {recovery_result['message']}\n"
            f"  目标: {device.ssh_user}@{device.ssh_host}:{device.ssh_port}\n"
            f"  命令: {device.recovery_command}\n"
            f"  恢复冷却期: {device.recovery_cooldown // 60} 分钟\n\n"
            f"恢复命令输出:\n"
            f"{recovery_result['output'] or '(无输出)'}\n\n"
            f"{'=' * 40}\n"
            f"可能原因:\n"
            f"  1. Termux 应用崩溃或被系统杀死\n"
            f"  2. 手机录音服务停止\n"
            f"  3. 网络/NAS 挂载异常\n\n"
            f"系统已按配置尝试自动恢复；若后续仍无新文件，请检查手机 Termux 状态。\n"
            f"{'=' * 40}\n"
            f"此告警冷却期: {device.stall_timeout // 60} 分钟\n"
            f"（同一问题不会在冷却期内重复发送）"
        )
        send_email_sync(subject, content)
        logger.info(f"📧 [{device.name}] 停滞告警邮件已发送")
    except Exception as e:
        logger.error(f"❌ [{device.name}] 发送停滞告警邮件失败: {e}")


def _send_stall_recovered_email(device, stall_active_since, last_file_time):
    try:
        from email_utils import send_email_sync

        now = datetime.now()
        stalled_minutes = max(0, int((last_file_time - stall_active_since) // 60))
        stall_start_str = datetime.fromtimestamp(stall_active_since).strftime("%Y-%m-%d %H:%M:%S")
        last_file_str = datetime.fromtimestamp(last_file_time).strftime("%Y-%m-%d %H:%M:%S")

        subject = f"✅ {device.name} 录音上传已恢复"
        content = (
            f"✅ {device.name} 录音上传已恢复\n"
            f"{'=' * 40}\n\n"
            f"恢复确认时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"停滞开始时间: {stall_start_str}\n"
            f"最新收到文件时间: {last_file_str}\n"
            f"本次停滞约: {stalled_minutes} 分钟\n"
            f"监控目录: {device.source_dir}\n\n"
            f"系统已经重新检测到新音频文件到达，上传链路恢复正常。\n"
            f"{'=' * 40}"
        )
        send_email_sync(subject, content)
        logger.info(f"📧 [{device.name}] 上传恢复邮件已发送")
    except Exception as e:
        logger.error(f"❌ [{device.name}] 发送上传恢复邮件失败: {e}")


def _device_watchdog(device):
    state = _device_states[device.name]

    logger.info(f"🐕 [{device.name}] 上传停滞看门狗已启动")
    logger.info(f"   停滞阈值: {device.stall_timeout}秒, "
                f"恢复冷却: {device.recovery_cooldown}秒, "
                f"活跃时段: {device.active_hour_start}:00 ~ {device.active_hour_end}:00")

    CHECK_INTERVAL = 60
    PROBE_TIMEOUT = 45           # 单次目录扫描超时（秒）
    CONFIRM_WINDOW = 180         # 告警/恢复前的复测观察窗口（秒）
    CONFIRM_INTERVAL = 20        # 复测间隔（秒）
    MOUNT_ALERT_COOLDOWN = 7200  # 挂载异常告警冷却（秒）

    while True:
        try:
            time.sleep(CHECK_INTERVAL)

            now = datetime.now()
            current_hour = now.hour

            if not (device.active_hour_start <= current_hour < device.active_hour_end):
                continue

            actual_latest = _get_latest_file_mtime_with_timeout(device, PROBE_TIMEOUT)
            if actual_latest > 0:
                with state.lock:
                    if actual_latest > state.last_new_file_time:
                        state.last_new_file_time = actual_latest

            with state.lock:
                elapsed = time.time() - state.last_new_file_time
                last_file_time = state.last_new_file_time

            if elapsed <= device.stall_timeout:
                stall_to_notify = 0.0
                with state.lock:
                    if state.stall_active_since > 0:
                        stall_to_notify = state.stall_active_since
                        state.stall_active_since = 0.0

                if stall_to_notify > 0:
                    _send_stall_recovered_email(device, stall_to_notify, last_file_time)

                continue

            elapsed_min = int(elapsed // 60)
            last_time_str = datetime.fromtimestamp(last_file_time).strftime("%Y-%m-%d %H:%M:%S")

            # ---- 挂载健康检查：挂载不可读 ≠ 手机上传停止 ----
            if not _probe_nas_mount(device.source_dir):
                logger.warning(f"📡 [{device.name}] NAS 挂载不可读（手机上传可能正常，是 Mac 侧挂载假死），"
                               "不触发上传停滞告警和手机重启，等待挂载看门狗修复")
                with state.lock:
                    mount_alert_due = (time.time() - state.last_mount_alert_time) >= MOUNT_ALERT_COOLDOWN
                    if mount_alert_due:
                        state.last_mount_alert_time = time.time()
                if mount_alert_due:
                    try:
                        from email_utils import send_email_sync
                        send_email_sync(
                            f"NAS 挂载异常（{device.name}）——非手机上传问题",
                            f"Mac 侧 SMB 挂载不可读，已 {elapsed_min} 分钟看不到 {device.name} 的新文件。\n"
                            "挂载看门狗会自动重挂（通常 1-3 分钟内恢复），手机上传不受影响。\n"
                            f"上次文件时间: {last_time_str}"
                        )
                    except Exception as e:
                        logger.error(f"[{device.name}] 挂载异常邮件发送失败: {e}")
                continue

            with state.lock:
                recovery_due = (time.time() - state.last_stall_recovery_time) >= device.recovery_cooldown
                alert_due = (time.time() - state.last_stall_alert_time) >= device.stall_timeout

            if not (recovery_due or alert_due):
                # 恢复与告警都在冷却期内：仅记录日志，等待下一轮
                logger.warning(f"🚨 [{device.name}] 上传停滞持续 {elapsed_min} 分钟 "
                               f"(上次文件: {last_time_str})，冷却等待中")
                continue

            logger.warning(f"🚨 [{device.name}] 上传停滞 {elapsed_min} 分钟 "
                           f"(上次文件: {last_time_str})，开始复测确认...")

            # ---- 复测：发告警/执行恢复前观察一小段时间，排除"正在自行恢复"的抖动 ----
            recheck_mtime = 0
            confirm_deadline = time.time() + CONFIRM_WINDOW
            while time.time() < confirm_deadline:
                time.sleep(CONFIRM_INTERVAL)
                recheck_mtime = _get_latest_file_mtime_with_timeout(device, PROBE_TIMEOUT)
                if recheck_mtime > last_file_time:
                    break

            if recheck_mtime > last_file_time:
                with state.lock:
                    if recheck_mtime > state.last_new_file_time:
                        state.last_new_file_time = recheck_mtime
                logger.info(f"👀 [{device.name}] 复测发现新文件已到达 "
                            f"({datetime.fromtimestamp(recheck_mtime).strftime('%H:%M:%S')})，"
                            f"判定为短暂停滞，跳过本次告警与恢复")
                continue

            # ---- 复测确认仍无新文件：真停滞，执行恢复与告警 ----
            with state.lock:
                if state.stall_active_since <= 0:
                    state.stall_active_since = last_file_time

            recovery_result = {
                "attempted": False,
                "ok": False,
                "message": "自动恢复冷却中，未重复执行",
                "output": "",
            }
            if recovery_due:
                with state.lock:
                    state.last_stall_recovery_time = time.time()
                recovery_result = _attempt_ssh_recovery(device, elapsed_min, last_time_str)

            if alert_due:
                if device.room != "unknown" and _room_has_other_active_device(device):
                    # ---- 房间视角降级：同房间其他设备仍在录音，房间覆盖正常 ----
                    # 为什么客厅两台都能降级：B轨有多源fallback(audio_processor)，
                    # Sony-2(分析源)停滞10分钟后自动切Sony-1接管分析，哭声检测不断流，
                    # 故单机停滞只等于"客厅降级单机运行"。两台都停时
                    # _room_has_other_active_device 返回False，走正常失聪警报。
                    # 卧室Sony-3是单点(room=bedroom无同伴)，其停滞永远正常警报。
                    # 只静默恢复+低频轻提醒，不更新告警冷却（若之后同房间全部停滞可立即升级为失聪警报）
                    room_label = ROOM_LABELS.get(device.room, device.room)
                    with state.lock:
                        throttle_due = (time.time() - state.last_room_degrade_log_time) >= 3600
                        if throttle_due:
                            state.last_room_degrade_log_time = time.time()
                    if throttle_due:
                        try:
                            from email_utils import send_email_sync
                            send_email_sync(
                                f"🟡 [{room_label}] {device.name} 录音停滞已自动处理（{room_label}覆盖正常）",
                                f"{device.name}（{room_label}）录音上传停滞 {elapsed_min} 分钟。\n\n"
                                f"同房间的其他设备仍在正常录音，{room_label}的听觉覆盖没有中断。\n"
                                f"系统已自动执行恢复命令，无需人工干预。\n"
                                f"上次收到文件: {last_time_str}"
                            )
                        except Exception as e:
                            logger.error(f"[{device.name}] 房间降级提醒邮件发送失败: {e}")
                    logger.warning(
                        f"🟡 [{device.name}] 上传停滞 {elapsed_min} 分钟，但同房间其他设备仍在录音"
                        f"（{room_label}覆盖正常）——已静默恢复，不发送失聪警报"
                    )
                else:
                    with state.lock:
                        state.last_stall_alert_time = time.time()
                    _send_stall_alert_email(device, elapsed_min, last_time_str, recovery_result)

        except Exception as e:
            logger.error(f"[{device.name}] 看门狗线程异常: {e}")


def _init_device_state(device):
    state = DeviceState()
    try:
        source_dir = device.source_dir
        latest_mtime = 0
        if os.path.exists(source_dir):
            supported_formats = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']
            for item in os.listdir(source_dir):
                if item in ["processed", "failed", "audio_segments", "logs"] or item.startswith('.'):
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
                        logger.error(f"[{device.name}] 读取子目录 {item_path} 失败: {e}")

                for filepath in items_to_check:
                    filename = os.path.basename(filepath)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_formats:
                        mtime = os.path.getmtime(filepath)
                        latest_mtime = max(latest_mtime, mtime)

        if latest_mtime > 0:
            state.last_new_file_time = latest_mtime
            logger.info(f"🐕 [{device.name}] 看门狗基准时间已初始化: "
                        f"{datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.info(f"🐕 [{device.name}] 看门狗基准时间: 当前时间（目录中无文件）")
    except Exception as e:
        logger.warning(f"⚠️ [{device.name}] 初始化看门狗基准时间失败: {e}")

    return state


def get_all_stall_status():
    result = {}
    for device in _devices:
        state = _device_states.get(device.name)
        if not state:
            continue
        with state.lock:
            elapsed = time.time() - state.last_new_file_time
            is_stalled = elapsed > device.stall_timeout
            result[device.name] = {
                "enabled": device.enabled,
                "ssh_host": device.ssh_host,
                "source_dir": device.source_dir,
                "room": device.room,
                "last_file_time": datetime.fromtimestamp(state.last_new_file_time).strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": int(elapsed),
                "is_stalled": is_stalled,
                "threshold_seconds": device.stall_timeout,
            }
    return result


def start_recovery_monitors():
    global _devices, _device_states, _initialized

    if _initialized:
        logger.info("📡 恢复监控已在运行中")
        return

    _devices = _parse_devices_from_env()

    if not _devices:
        logger.info("📡 未配置恢复监控设备，跳过启动")
        _initialized = True
        return

    for device in _devices:
        if not device.enabled:
            logger.info(f"📡 [{device.name}] 设备已禁用，跳过")
            continue
        state = _init_device_state(device)
        _device_states[device.name] = state

        t = threading.Thread(target=_device_watchdog, args=(device,), daemon=True)
        t.start()

    _initialized = True
    logger.info(f"📡 恢复监控已启动，共 {len([d for d in _devices if d.enabled])} 个启用设备")


def trigger_recovery(device_name):
    device = None
    for d in _devices:
        if d.name == device_name:
            device = d
            break

    if not device:
        return {"success": False, "message": f"设备 {device_name} 不存在"}

    if not device.enabled:
        return {"success": False, "message": f"设备 {device_name} 未启用"}

    state = _device_states.get(device_name)
    if not state:
        return {"success": False, "message": f"设备 {device_name} 状态未初始化"}

    with state.lock:
        elapsed = time.time() - state.last_new_file_time
        last_time_str = datetime.fromtimestamp(state.last_new_file_time).strftime("%Y-%m-%d %H:%M:%S")
        elapsed_min = int(elapsed // 60)

    result = _attempt_ssh_recovery(device, elapsed_min, last_time_str)

    with state.lock:
        state.last_stall_recovery_time = time.time()

    return {
        "success": result["ok"],
        "attempted": result["attempted"],
        "message": result["message"],
        "output": result["output"],
    }
