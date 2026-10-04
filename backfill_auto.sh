#!/bin/bash
# 凌晨自动补跑调度（launchd com.asr.backfill 每天 01:05 触发）
# 流程：检查 5008 存活 → 暂停 B 轨实时转写（让出 GPU）→ 跑 backfill_5to9.sh
#       （断点续传 + 凌晨窗口硬约束 + 挂载守卫）→ 无论成败恢复 B 轨
# 手动补跑仍可用手机端「启动」按钮（走 5008 API，B 轨同样自动让路）。
cd /Users/mac/asr-server || exit 1
LOG=log/backfill_auto.log
ts() { date '+%m-%d %H:%M:%S'; }

# ADMIN_TOKEN 全文读取（勿截断）
TOKEN=$(grep '^ADMIN_TOKEN=' .env | head -1 | cut -d= -f2-)
ASR=http://127.0.0.1:5008

restore_b() {
    /usr/bin/curl -s -X POST -H "X-Admin-Token: $TOKEN" "$ASR/api/start_live" >/dev/null 2>&1
    echo "[$(ts)] ✅ B 轨实时转写已恢复" >> "$LOG"
}
trap restore_b EXIT

# 前置检查：5008 必须存活（挂载/服务问题交给各自的 watchdog，不越权处理）
if ! /usr/bin/curl -s -H "X-Admin-Token: $TOKEN" "$ASR/manage" >/dev/null 2>&1; then
    echo "[$(ts)] ⚠️ 5008 未响应，跳过本次调度（服务/挂载看门狗会处理）" >> "$LOG"
    exit 0
fi

# 已有补跑在跑（手机端手动启动的）就不重复触发
if pgrep -f "reprocess_history_cries.py" >/dev/null 2>&1; then
    echo "[$(ts)] ℹ️ 检测到补跑已在运行，本次不重复启动" >> "$LOG"
    exit 0
fi

echo "[$(ts)] ########## 凌晨自动补跑启动 ##########" >> "$LOG"
/usr/bin/curl -s -X POST -H "X-Admin-Token: $TOKEN" "$ASR/api/pause_live" >/dev/null 2>&1
echo "[$(ts)] B 轨已暂停，GPU 让给补跑" >> "$LOG"

bash backfill_5to9.sh >> log/backfill_5to9.log 2>&1
echo "[$(ts)] backfill_5to9.sh 退出码=$?" >> "$LOG"
# EXIT trap 会自动恢复 B 轨
