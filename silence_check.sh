#!/bin/bash
# 静音巡检：扫录音设备今日最新文件（2026-10-03 起仅双 Pixel；Sony 已停录移出巡检）
#   - 全零静音（max_volume < -80dB）→ 告警（重装/重启后未点开 App，麦克风豁免失效）
#   - 超过 6 分钟无新文件 → 告警（设备掉线 / 服务未启动）
# 背景：2026-10-03 48k 码率全零事故 + 锁屏后台启动全零（Sony 系 RECORD_AUDIO=foreground）
cd /Users/mac/asr-server

RECORDS=/Volumes/download/records
TODAY=$(date +%F)
DEVICES=(Pixel-6 Pixel-5)
ALERT_STATE=/tmp/silence_check_alert_ts
COOLDOWN=1800
issues=()

for dev in "${DEVICES[@]}"; do
  dir="$RECORDS/$dev/$TODAY"
  latest=$(ls -t "$dir"/*.m4a 2>/dev/null | head -1)
  if [ -z "$latest" ]; then
    issues+=("$dev: 今日目录无录音文件")
    continue
  fi
  age=$(( ( $(date +%s) - $(stat -f %m "$latest") ) / 60 ))
  vol=$(ffmpeg -i "$latest" -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | awk '{print $1}')
  if [ "$age" -ge 6 ]; then
    issues+=("$dev: ${age} 分钟无新录音（最新: $(basename "$latest")）")
  elif [ -n "$vol" ] && awk -v v="$vol" 'BEGIN{exit !(v < -80)}'; then
    issues+=("$dev: 最新录音全零静音（${vol} dB, $(basename "$latest")）—— 疑似重装/重启后未点开 App")
  fi
done

[ ${#issues[@]} -eq 0 ] && exit 0

# 告警冷却
now=$(date +%s)
last=$(cat "$ALERT_STATE" 2>/dev/null || echo 0)
[ $((now - last)) -lt $COOLDOWN ] && exit 0
echo "$now" > "$ALERT_STATE"

body=$(printf '%s\n' "${issues[@]}")
/bin/bash -c "python3 - <<'PYEOF'
from email_utils import send_email_sync
send_email_sync('录音设备静音告警', '''录音巡检发现异常：

$body

处理提示：重装 APK 或手机重启后，需解锁屏幕并点开一次录音 App
（Android 14 仅前台麦克风模式，App 前台过一次即永久豁免）。
''')
PYEOF" 2>/dev/null || echo "$(date '+%F %T') ⚠️ 邮件发送失败"
