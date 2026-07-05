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
                 active_hour_start=6, active_hour_end=23):
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


class DeviceState:
    def __init__(self):
        self.last_new_file_time = time.time()
        self.last_stall_alert_time = 0.0
        self.last_stall_recovery_time = 0.0
        self.stall_active_since = 0.0
        self.lock = threading.Lock()


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
        )
        devices.append(device)
        logger.info(f"📱 已配置恢复监控设备: {name} ({ssh_host}) -> {source_dir}")

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

    while True:
        try:
            time.sleep(CHECK_INTERVAL)

            now = datetime.now()
            current_hour = now.hour

            if not (device.active_hour_start <= current_hour < device.active_hour_end):
                continue

            actual_latest = _get_latest_file_mtime(device)
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

            with state.lock:
                if state.stall_active_since <= 0:
                    state.stall_active_since = last_file_time

            logger.warning(f"🚨 [{device.name}] 上传停滞告警！已 {elapsed_min} 分钟没有新文件到达 "
                           f"(上次文件: {last_time_str})")

            recovery_result = {
                "attempted": False,
                "ok": False,
                "message": "自动恢复冷却中，未重复执行",
                "output": "",
            }
            with state.lock:
                if (time.time() - state.last_stall_recovery_time) >= device.recovery_cooldown:
                    state.last_stall_recovery_time = time.time()
                    do_recover = True
                else:
                    do_recover = False

            if do_recover:
                recovery_result = _attempt_ssh_recovery(device, elapsed_min, last_time_str)

            with state.lock:
                if (time.time() - state.last_stall_alert_time) < device.stall_timeout:
                    continue
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
