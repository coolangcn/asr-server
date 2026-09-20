#!/bin/bash
# 补跑 2026-05~09 漏检文件（用 2026-09-18 校准的新规则）
# 顺序：先 7/8/9 月（旧规则全漏检窗口），再 5/6 月（旧规则边缘漏检）
# 支持断点续传：reprocess_history_cries.py 内部会跳过已确认哭声的文件，可随时中断重跑
cd /Users/mac/asr-server || exit 1
PROC=/Volumes/download/records/Sony-2/processed
PY=/Users/mac/asr_env/bin/python3
LOG=backfill_5to9.log

# 带超时的 ls（SMB 挂载假死时 ls 会无限期挂起，卡死整个补跑）
safe_ls_count() {
    ls "$1" 2>/dev/null &
    local pid=$!
    ( sleep 50; kill -9 $pid 2>/dev/null ) & local killer=$!
    # 轮询等待：SMB 卡死是 D 状态，kill -9 打不断，wait 会无限等——超时放弃
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 50 ]; do
        sleep 1; waited=$((waited+1))
    done
    kill $killer 2>/dev/null; wait $killer 2>/dev/null
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 $pid 2>/dev/null
        echo "TIMEOUT"; return
    fi
    wait $pid 2>/dev/null
    local rc=$?
    [ $rc -ne 0 ] && { echo "TIMEOUT"; return; }
    ls "$1" 2>/dev/null | wc -l | tr -d ' '
}

run_date() {
    local d="$1"
    local n
    n=$(safe_ls_count "$PROC/$d")
    if [ "$n" = "TIMEOUT" ]; then
        echo "[$(date '+%m-%d %H:%M:%S')] ⚠️ ls $d 超时(SMB 异常)，等 60s 后由外层重试"
        sleep 60
        n=$(safe_ls_count "$PROC/$d")
        if [ "$n" = "TIMEOUT" ]; then
            echo "[$(date '+%m-%d %H:%M:%S')] ❌ $d 两次超时，跳过该日期"
            return
        fi
    fi
    [ "$n" -eq 0 ] && { echo "[$(date '+%m-%d %H:%M:%S')] 跳过 $d (无文件)"; return; }
    echo "[$(date '+%m-%d %H:%M:%S')] === 补跑 $d ($n 个文件) ==="
    "$PY" reprocess_history_cries.py "$d" >> "$LOG" 2>&1
    echo "[$(date '+%m-%d %H:%M:%S')] === $d 完成 (exit=$?) ==="
}

echo "[$(date '+%m-%d %H:%M:%S')] ########## 补跑任务启动 ##########"

# 外层日期列表也做重试保护（挂载抖动时 ls 可能返回空，静默空跑=假完成）
get_dates() {
    local pattern="$1"
    local dates=""
    for attempt in 1 2 3; do
        dates=$(ls "$PROC" 2>/dev/null | grep -E "$pattern" | sort)
        [ -n "$dates" ] && { echo "$dates"; return 0; }
        echo "[$(date '+%m-%d %H:%M:%S')] ⚠️ 日期列表为空(第${attempt}次，挂载抖动?)，30s 后重试" >> "$LOG"
        sleep 30
    done
    return 1
}

if DATES=$(get_dates "^2026-(07|08|09)-"); then
    for d in $DATES; do run_date "$d"; done
else
    echo "[$(date '+%m-%d %H:%M:%S')] ❌ 7-9月日期列表三次获取失败，退出（可随时重跑续传）" >> "$LOG"
    exit 1
fi

if DATES=$(get_dates "^2026-(05|06)-"); then
    for d in $DATES; do run_date "$d"; done
else
    echo "[$(date '+%m-%d %H:%M:%S')] ❌ 5-6月日期列表三次获取失败，退出（可随时重跑续传）" >> "$LOG"
    exit 1
fi

echo "[$(date '+%m-%d %H:%M:%S')] ########## 全部补跑完成 ##########"
