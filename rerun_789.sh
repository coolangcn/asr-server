#!/bin/bash
# 【2026-09-26】7/8/9 月全量重检驱动 v3
# v2 教训: ① bash3.2 的 {02..31} 不补零 → 用 seq -w
#         ② --replace 会被当成 argv[2](开始时间) → 必须写 "$d" "" "" --replace
# 日志: log/rerun_789.log
PY=/Users/mac/asr_env/bin/python3
cd /Users/mac/asr-server
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> log/rerun_789.log; }
S2=/Volumes/download/records/Sony-2
S1=/Volumes/download/records/Sony-1
has_data() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

log "===== 7/8/9 月重检启动 (v3) ====="

# --- Sony-2 主扫 ---
for i in $(seq -w 2 31); do
    d="2026-07-$i"
    if ! has_data $S2/processed/$d && ! has_data $S2/$d; then log "--- 跳过 $d (无数据)"; continue; fi
    log "--- 开始 $d (Sony-2)"
    $PY -u reprocess_history_cries.py "$d" "" "" --replace >> log/rerun_789.log 2>&1
    log "--- 完成 $d (exit=$?)"
done
for i in $(seq -w 1 31); do
    d="2026-08-$i"
    if ! has_data $S2/processed/$d && ! has_data $S2/$d; then log "--- 跳过 $d (无数据)"; continue; fi
    log "--- 开始 $d (Sony-2)"
    $PY -u reprocess_history_cries.py "$d" "" "" --replace >> log/rerun_789.log 2>&1
    log "--- 完成 $d (exit=$?)"
done
for i in $(seq -w 1 8); do
    d="2026-09-0$i"
    if ! has_data $S2/processed/$d && ! has_data $S2/$d; then log "--- 跳过 $d (无数据)"; continue; fi
    log "--- 开始 $d (Sony-2)"
    $PY -u reprocess_history_cries.py "$d" "" "" --replace >> log/rerun_789.log 2>&1
    log "--- 完成 $d (exit=$?)"
done
for d in 2026-09-25 2026-09-26; do
    if ! has_data $S2/processed/$d && ! has_data $S2/$d; then log "--- 跳过 $d (无数据)"; continue; fi
    log "--- 开始 $d (Sony-2)"
    $PY -u reprocess_history_cries.py "$d" "" "" --replace >> log/rerun_789.log 2>&1
    log "--- 完成 $d (exit=$?)"
done

# --- Sony-1 八月积压（环境变量切源）---
for i in $(seq -w 1 31); do
    d="2026-08-$i"
    if ! has_data $S1/processed/$d && ! has_data $S1/$d; then log "--- 跳过 $d (Sony-1 无数据)"; continue; fi
    log "--- 开始 $d (Sony-1)"
    REPROCESS_SOURCE_DIR=/Volumes/download/records/Sony-1 \
        $PY -u reprocess_history_cries.py "$d" "" "" --replace >> log/rerun_789.log 2>&1
    log "--- 完成 $d Sony-1 (exit=$?)"
done

log "===== 7/8/9 月重检全部完成 ====="
