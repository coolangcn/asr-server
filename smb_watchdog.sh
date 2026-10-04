#!/bin/bash
# ==============================================
#   SMB 挂载看门狗 v4 (launchd 每 90 秒调用)
#   设计原则：宁可漏判，不可误杀
#   - 探测超时 60 秒 + 连续 5 轮(约7.5分钟)失败才动手：
#     NAS 高负载(补跑+5008并发)时阵发慢是常态，探测超时≠挂载死亡
#     (2026-09-19 v3 曾因 30s+2轮 过于激进，一天误拆健康挂载 13 次)
#   - 仅用 mount_smbfs 挂载，Finder/osascript 通道已弃用：
#     当天 mount volume / open smb:// 集体挂起且可能弹密码框卡死后台
#   - 目录枚举探测：SMB 会话半损坏时"列表截断"(挂载可见但条目骤减且
#     稳定不自行恢复)，读单个文件探测发现不了，必须数 records/ 条目数
#   - 超时用轮询等待而非 wait：SMB 卡死时子进程进 D 状态(不可中断)，
#     kill -9 都打不死，传统 wait 会跟着无限等(2026-09-20 看门狗睡死7.7h教训)
# ==============================================

MOUNT_POINT="/Volumes/download"
SMB_URL="smb://admin:74123698cN@192.168.1.188/download"
SMB_URL_CLI="//admin:74123698cN@192.168.1.188/download"
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
    # 【2026-09-20】重大发现：launchd 上下文中 bash/ls 访问 SMB 挂载被 macOS TCC 拦截
    # (Operation not permitted，瞬时失败)，而 python3 二进制持有网络卷授权——
    # 因此探测必须用 python3 而非 ls。此前探测 100% 假失败，导致看门狗每 7.5 分钟
    # 强拆重挂一次健康挂载（截断病反复发作的重大嫌疑）。
    # 注：mount_smbfs/unmount 走内核 syscall 不受 TCC 限制，无需更换。
    with_timeout $PROBE_TIMEOUT /Users/mac/asr_env/bin/python3 -c "import os; os.listdir('$CHECK_PATH')" > /dev/null 2>&1 || return 1
    # 目录枚举探测：SMB 会话半损坏时会"列表截断"（能看到挂载但条目骤减，
    # 且截断状态稳定不恢复）。records/ 正常有 Sony-1/2/3 等条目，
    # 少于 3 个即判定为病态挂载，走重挂流程。
    local entries
    entries=$(with_timeout $PROBE_TIMEOUT /Users/mac/asr_env/bin/python3 -c "import os; print(len(os.listdir('$MOUNT_POINT/records')))" 2>/dev/null | tr -d ' \r')
    [ -n "$entries" ] && [ "$entries" -ge 3 ]
}

is_mounted() {
    mount | grep -q "on $MOUNT_POINT"
}

# ---- B 轨监听假死检测（2026-10-03）----
# 病：挂载健康，但 5008 监听线程的内核 SMB 调用悬死（不可中断等待），
#     监听"活着"却永远扫不到新文件。普通重启进程无效，必须强制重挂清内核脏会话。
# 信号：5008 每轮扫描刷新 log/b_track_heartbeat 的 mtime（连续 SMB 超时 180s
#     时刻意停滞心跳）。mtime 超过 B_HEARTBEAT_MAX_AGE 即判定假死。
# 动作：杀 5008 → 强制重挂（清内核脏状态）→ 拉起 → 重置心跳。
B_HEARTBEAT="/Users/mac/asr-server/log/b_track_heartbeat"
B_HEARTBEAT_MAX_AGE=240
BTRACK_ALERT_STATE="/tmp/smb_watchdog_btrack_alert_ts"

btrack_selfheal() {
    local now=$(date +%s)
    local last=$(cat "$BTRACK_ALERT_STATE" 2>/dev/null || echo 0)
    if [ $((now - last)) -ge $ALERT_COOLDOWN ]; then
        echo "$now" > "$BTRACK_ALERT_STATE"
        /Users/mac/asr_env/bin/python3 -c "
import sys
sys.path.insert(0, '/Users/mac/asr-server')
from email_utils import send_email_sync
send_email_sync('B 轨监听假死已自动恢复', '''检测到 5008 B 轨监听心跳停滞（内核 SMB 会话悬死），
已自动执行：杀 5008 → 强制重挂 SMB → 拉起服务。
积压录音将由监听自动追赶，无需人工干预。
详情: $LOG_FILE''')
" 2>/dev/null || log "⚠️ 邮件发送失败"
    fi
    log "🚨 [B轨心跳] 停滞 ${age}s，启动自愈: 杀5008 → 重挂 → 拉起"
    pkill -f "asr_server.py" 2>/dev/null
    sleep 3
    with_timeout $UNMOUNT_TIMEOUT diskutil unmount force "$MOUNT_POINT" >> "$LOG_FILE" 2>&1
    sleep 2
    if ! is_mounted; then
        with_timeout 45 mount_smbfs "$SMB_URL_CLI" "$MOUNT_POINT" >> "$LOG_FILE" 2>&1 || log "⚠️ 重挂失败（下一轮看门狗继续处理挂载）"
    fi
    # 拉起服务：kickstart 仅对已加载服务有效（bootout 后找不到），失败则 bootstrap 兜底
    launchctl kickstart -k gui/501/com.asr.server 2>>"$LOG_FILE" || \
        launchctl bootstrap gui/501 "$HOME/Library/LaunchAgents/com.asr.server.plist" >> "$LOG_FILE" 2>&1
    touch "$B_HEARTBEAT"   # 给新进程宽限期（模型加载 ~60s 后监听线程接管心跳）
    log "✅ [B轨心跳] 自愈动作完成（杀5008+重挂+拉起），等待新进程接管心跳"
}

check_btrack_heartbeat() {
    # 心跳文件不存在 = 监听从未跑过或刚清过日志，不触发
    [ -f "$B_HEARTBEAT" ] || return 0
    age=$(( $(date +%s) - $(stat -f %m "$B_HEARTBEAT" 2>/dev/null || date +%s) ))
    [ "$age" -le "$B_HEARTBEAT_MAX_AGE" ] && return 0
    btrack_selfheal
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
    check_btrack_heartbeat   # 挂载健康但监听线程 SMB 悬死 → 假死自愈
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
