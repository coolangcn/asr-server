#!/bin/bash
# 凌晨自动补跑调度（launchd com.asr.backfill 每天 01:05 触发）
# 流程：检查 5008 存活 → 跑 backfill_5to9.sh（断点续传 + 凌晨窗口硬约束 + 挂载守卫）
# 【2026-10-04 夜间降级】不再 pause_live 全停 B 轨：B 轨对凌晨录音自动降级为
#        仅哭声检测（audio_processor skip_asr，GPU 占用极小），半夜哭声告警不再盲区；
#        A 轨补跑独占转写算力不变。手动补跑仍可用手机端「启动」按钮。
cd /Users/mac/asr-server || exit 1
LOG=log/backfill_auto.log
ts() { date '+%m-%d %H:%M:%S'; }
ASR=http://127.0.0.1:5008

# 前置检查：5008 必须存活（挂载/服务问题交给各自的 watchdog，不越权处理）
if ! /usr/bin/curl -s -H "X-Admin-Token: $(grep '^ADMIN_TOKEN=' .env | head -1 | cut -d= -f2-)" "$ASR/manage" >/dev/null 2>&1; then
    echo "[$(ts)] ⚠️ 5008 未响应，跳过本次调度（服务/挂载看门狗会处理）" >> "$LOG"
    exit 0
fi

# 已有补跑在跑（手机端手动启动的）就不重复触发
if pgrep -f "reprocess_history_cries.py" >/dev/null 2>&1; then
    echo "[$(ts)] ℹ️ 检测到补跑已在运行，本次不重复启动" >> "$LOG"
    exit 0
fi

echo "[$(ts)] ########## 凌晨自动补跑启动 ##########" >> "$LOG"
echo "[$(ts)] B 轨处于夜间降级模式（仅哭声检测），GPU 主要让给补跑" >> "$LOG"

bash backfill_5to9.sh >> log/backfill_5to9.log 2>&1
echo "[$(ts)] backfill_5to9.sh 退出码=$?" >> "$LOG"
