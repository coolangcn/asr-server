#!/bin/bash
# 【2026-09-27】7/8/9 月重检 v4：只跑剩余未处理日期
# 已处理(跳过): 07-02,03,04,09,14,15（当时缓存是全量的，真实检测过，0哭声）
# 修复: reprocess 跳过缓存快捷路径(--replace 定向模式)，每次重扫当天目录
# 日志: log/rerun_789_v4.log
PY=/Users/mac/asr_env/bin/python3
cd /Users/mac/asr-server
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> log/rerun_789_v4.log; }
S2=/Volumes/download/records/Sony-2
S1=/Volumes/download/records/Sony-1
has_data() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

log "===== 7/8/9 月重检 v4 启动（剩余日期） ====="

for d in 2026-07-{05,06,07,08,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31} \
         2026-08-{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31} \
         2026-09-{01,02,03,04,05,06,07,08,25,26}; do
    if ! has_data $S2/processed/$d && ! has_data $S2/$d; then log "--- 跳过 $d (无数据)"; continue; fi
    log "--- 开始 $d (Sony-2)"
    $PY -u reprocess_history_cries.py "$d" "" "" --replace >> log/rerun_789_v4.log 2>&1
    log "--- 完成 $d (exit=$?)"
done

for d in 2026-08-{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}; do
    if ! has_data $S1/processed/$d && ! has_data $S1/$d; then log "--- 跳过 $d (Sony-1 无数据)"; continue; fi
    log "--- 开始 $d (Sony-1)"
    REPROCESS_SOURCE_DIR=/Volumes/download/records/Sony-1 \
        $PY -u reprocess_history_cries.py "$d" "" "" --replace >> log/rerun_789_v4.log 2>&1
    log "--- 完成 $d Sony-1 (exit=$?)"
done

log "===== v4 全部完成 ====="
