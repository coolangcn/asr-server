#!/bin/bash
# ==============================================
#   SMB 挂载看门狗 v3 (launchd 每 90 秒调用)
#   设计原则：宁可漏判，不可误杀
#   - 探测超时 30 秒（补跑高负载时 NAS 响应慢是正常的）
#   - 连续 2 轮探测失败才动手（防单次慢读误判）
#   - 用 Finder 正规途径挂载（osascript），不走裸 mount_smbfs
#     （mount_smbfs 直接挂 /Volumes 会被磁盘仲裁卡掉）
#   - 2026-09-19 v2 教训：误判健康挂载并强拆，造成反复断挂
# ==============================================

MOUNT_POINT="/Volumes/download"
SMB_URL="smb://admin:cncncncn@192.168.1.188/download"
SMB_URL_CLI="//admin:cncncncn@192.168.1.188/download"
CHECK_PATH="/Volumes/download/records"
PROBE_TIMEOUT=60
UNMOUNT_TIMEOUT=30
FAILS_NEEDED=5
LOG_FILE="/Users/mac/asr-server/log/smb_watchdog.log"
STATE_FILE="/tmp/smb_watchdog_fails"
ALERT_STATE="/tmp/smb_watchdog_alert_ts"
LOCK_DIR="/tmp/smb_watchdog.lock"
ALERT_COOLDOWN=1800

cd /Users/mac/asr-server
mkdir "$LOCK_DIR" 2>/dev/null || exit 0
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

send_alert() {
    local now=$(date +%s)
    local last=$(cat "$ALERT_STATE" 2>/dev/null || echo 0)
    [ $((now - last)) -lt $ALERT_COOLDOWN ] && return
    echo "$now" > "$ALERT_STATE"
    /Users/mac/asr_env/bin/python3 -c "
import sys
sys.path.insert(0, '/Users/mac/asr-server')
from email_utils import send_email_sync
send_email_sync('NAS 挂载自动重挂失败', '''SMB 看门狗连续重挂失败：
目标: $SMB_URL
日志: $LOG_FILE
录音文件仍在上传到 NAS（不受影响），但 Mac 侧暂时无法读取处理。
请检查 NAS SMB 服务。''')
" 2>/dev/null || log "⚠️ 邮件发送失败"
}

with_timeout() {
    local t="$1"; shift
    "$@" &
    local pid=$!
    ( sleep "$t"; kill -9 $pid 2>/dev/null ) & local killer=$!
    # 轮询等待：SMB 卡死时子进程处于 D 状态（不可中断），kill -9 无法终止，
    # 传统 wait 会无限等下去拖死整个看门狗 —— 超时后必须放弃等待继续走
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$t" ]; do
        sleep 1; waited=$((waited+1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null
        kill "$killer" 2>/dev/null
        return 124
    fi
    wait $pid 2>/dev/null
    local rc=$?
    kill $killer 2>/dev/null
    wait $killer 2>/dev/null
    return $rc
}

probe_alive() {
    with_timeout $PROBE_TIMEOUT ls "$CHECK_PATH" > /dev/null 2>&1 || return 1
    # 目录枚举探测：SMB 会话半损坏时会"列表截断"（能看到挂载但条目骤减，
    # 且截断状态稳定不恢复）。records/ 正常有 Sony-1/2/3 + processed 等条目，
    # 少于 3 个即判定为病态挂载，走重挂流程。
    local entries
    entries=$(with_timeout $PROBE_TIMEOUT ls "$MOUNT_POINT/records" 2>/dev/null | wc -l | tr -d ' ')
    [ -n "$entries" ] && [ "$entries" -ge 3 ]
}

is_mounted() {
    mount | grep -q "on $MOUNT_POINT"
}

finder_mount() {
    # 仅用 mount_smbfs（CLI 通道稳定）；Finder/osascript 通道会挂起且可能弹密码框，已弃用
    log "🔗 mount_smbfs 直接挂载..."
    if [ -d "$MOUNT_POINT" ] && with_timeout 45 mount_smbfs "$SMB_URL_CLI" "$MOUNT_POINT" >> "$LOG_FILE" 2>&1; then
        sleep 2
        if probe_alive; then
            log "✅ mount_smbfs 挂载成功且探测通过"
            return 0
        fi
        log "⚠️ mount_smbfs 挂载后探测未通过"
    fi
    log "❌ mount_smbfs 挂载失败（等待下一轮重试）"
    return 1
}

# ---- 主流程 ----
if probe_alive; then
    if [ -s "$STATE_FILE" ] && [ "$(cat "$STATE_FILE")" -ge "$FAILS_NEEDED" ]; then
        log "✅ 挂载已恢复（此前累计失败 $(cat "$STATE_FILE") 轮）"
    fi
    echo 0 > "$STATE_FILE"
    exit 0
fi

# 第一轮失败：计数，不动手
FAILS=$(( $(cat "$STATE_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$FAILS" > "$STATE_FILE"
log "⚠️ 探测失败（连续第 $FAILS 轮，需 $FAILS_NEEDED 轮才动手）"
[ "$FAILS" -lt "$FAILS_NEEDED" ] && exit 0

# ---- 连续 $FAILS_NEEDED 轮失败，开始修复 ----
log "🛠️ 连续 $FAILS 轮探测失败，开始修复..."

if is_mounted; then
    log "🧹 挂载记录存在但不可读，先卸载..."
    if ! with_timeout $UNMOUNT_TIMEOUT diskutil unmount force "$MOUNT_POINT" >> "$LOG_FILE" 2>&1; then
        log "❌ 卸载失败/超时，本轮放弃（下轮再试）"
        send_alert
        exit 1
    fi
    sleep 2
fi

# 注意：不做 rmdir —— /Volumes 下用户无权删目录（root 才行，此步历来无效），
# 且会把正常挂载点删掉；挂载点丢失由 com.asr.mountkeeper 守护（root，每 5 分钟确保存在）

if finder_mount; then
    echo 0 > "$STATE_FILE"
else
    send_alert
    exit 1
fi
