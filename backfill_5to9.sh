#!/bin/bash
# 补跑 2026-05~09 漏检文件（用 2026-09-18 校准的新规则）v2
# v2 防挂载截断病（2026-09-20 教训：上次补跑在病态挂载期"假完成"，
#   日期列表被截断(8月整月丢失) + exists() 误报导致 18365 个文件被跳过，真实检测率<1%）
# 三层防护：
#   1. 启动时挂载健康验证 + 日期快照到本地（后续不再依赖中途 ls 的完整性）
#   2. 每个日期跑之前做"日期完整性守卫"（可见日期数 < 快照85% → 等待/中止）
#   3. reprocess 检测到文件大面积不可达时 exit 3 → 外层中止，可随时重跑断点续传
# 断点续传：reprocess 内部只跳过 status='cry'（已确认哭声）的文件，其余全部重检
cd /Users/mac/asr-server || exit 1
PROC=/Volumes/download/records/Sony-2/processed
PY=/Users/mac/asr_env/bin/python3
LOG=backfill_5to9.log
SNAP=/Users/mac/asr-server/backfill_dates_snapshot.txt
FAILED=/Users/mac/asr-server/backfill_failed_dates.txt

# 带超时的 ls（SMB 挂载假死时 ls 会无限期挂起，卡死整个补跑）
# 注意：探测用 ls 的 stdout 必须丢弃（只要退出码），否则在 $() 里运行时
# 212 行目录名会混进返回值，数字比较永远失败（2026-09-20 v2 启动卡死根因）
safe_ls_count() {
    ls "$1" > /dev/null 2>&1 &
    local pid=$!
    ( sleep 50; kill -9 $pid 2>/dev/null ) & local killer=$!
    wait $pid 2>/dev/null
    local rc=$?
    kill $killer 2>/dev/null; wait $killer 2>/dev/null
    [ $rc -ne 0 ] && { echo "TIMEOUT"; return; }
    ls "$1" 2>/dev/null | wc -l | tr -d ' '
}

# ====== 启动：挂载健康验证 + 日期快照 ======
echo "[$(date '+%m-%d %H:%M:%S')] ########## 补跑任务启动 (v2 防截断病版) ##########"
total_dates=0
for i in 1 2 3; do
    n=$(safe_ls_count "$PROC")
    if [ "$n" != "TIMEOUT" ] && [ "$n" -ge 130 ]; then
        total_dates=$n
        break
    fi
    echo "[$(date '+%m-%d %H:%M:%S')] ⚠️ processed 可见日期数=$n (期望≥130，疑似截断病)，90s 后重试 ($i/3)"
    sleep 90
done
if [ "$total_dates" -lt 130 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] ❌ 挂载日期列表持续不完整，放弃启动（防止假完成）" | tee -a "$LOG"
    exit 4
fi
ls "$PROC" 2>/dev/null | grep -E '^2026-(0[5-9])-' | sort > "$SNAP"
snap_n=$(wc -l < "$SNAP" | tr -d ' ')
echo "[$(date '+%m-%d %H:%M:%S')] ✅ 挂载健康（processed 可见 $total_dates 个目录），日期快照 $snap_n 天 (5-9月) 已存 $SNAP"

# 日期完整性守卫：当前可见日期数 < 快照85% 视为截断病发作，等待重试，仍不行则中止
guard_mount() {
    local expect=$(( snap_n * 85 / 100 ))
    for i in 1 2 3; do
        local n=$(safe_ls_count "$PROC")
        if [ "$n" != "TIMEOUT" ] && [ "$n" -ge "$expect" ]; then
            return 0
        fi
        echo "[$(date '+%m-%d %H:%M:%S')] 🚨 守卫：可见目录数=$n (快照 $snap_n 的85%=$expect)，疑似截断病，90s 后重试 ($i/3)"
        sleep 90
    done
    return 1
}

run_date() {
    local d="$1"
    guard_mount || { echo "[$(date '+%m-%d %H:%M:%S')] ❌ $d: 挂载守卫连续失败，中止补跑 (exit 3)" | tee -a "$LOG"; echo "$d" >> "$FAILED"; exit 3; }
    local n
    n=$(safe_ls_count "$PROC/$d")
    if [ "$n" = "TIMEOUT" ]; then
        echo "[$(date '+%m-%d %H:%M:%S')] ⚠️ ls $d 超时(SMB 异常)，等 60s 后由外层重试"
        sleep 60
        n=$(safe_ls_count "$PROC/$d")
        if [ "$n" = "TIMEOUT" ]; then
            echo "[$(date '+%m-%d %H:%M:%S')] ❌ $d 两次超时，记入重试清单"
            echo "$d" >> "$FAILED"
            return
        fi
    fi
    [ "$n" -eq 0 ] && { echo "[$(date '+%m-%d %H:%M:%S')] 跳过 $d (无文件)"; return; }
    echo "[$(date '+%m-%d %H:%M:%S')] === 补跑 $d ($n 个文件) ==="
    "$PY" reprocess_history_cries.py "$d" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 3 ]; then
        echo "[$(date '+%m-%d %H:%M:%S')] 🚨 $d: reprocess 报告挂载异常 (exit 3)，中止整个补跑" | tee -a "$LOG"
        echo "$d" >> "$FAILED"
        exit 3
    elif [ $rc -ne 0 ]; then
        echo "[$(date '+%m-%d %H:%M:%S')] ⚠️ $d 异常退出 (exit=$rc)，记入重试清单"
        echo "$d" >> "$FAILED"
    else
        echo "[$(date '+%m-%d %H:%M:%S')] === $d 完成 (exit=0) ==="
    fi
}

# ====== 窗口硬约束：A 轨只许在凌晨 01:00-06:30 运行，白天永不运行（2026-10-03）======
# 每个日期开跑前检查；单日期跑到一半超窗不强杀（进度断点续传，明日接着跑）
window_check() {
    local h=$(date +%H%M)
    [ "$h" -ge 0100 ] && [ "$h" -lt 0630 ] && return 0
    echo "[$(date '+%m-%d %H:%M:%S')] ⏰ 超出凌晨窗口 (01:00-06:30)，补跑暂停，明日凌晨自动继续"
    exit 0
}

# ====== 主流程：7/8/9 月 → 5/6 月，按快照执行 ======
> "$FAILED"
for d in $(grep -E '^2026-(07|08|09)-' "$SNAP"); do window_check; run_date "$d" || break; done
RC1=$?
if [ $RC1 -ne 0 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] ❌ 7-9月阶段中止 (exit=$RC1)，可重跑本脚本断点续传" | tee -a "$LOG"
    exit $RC1
fi

for d in $(grep -E '^2026-(05|06)-' "$SNAP"); do window_check; run_date "$d" || break; done
RC2=$?
if [ $RC2 -ne 0 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] ❌ 5-6月阶段中止 (exit=$RC2)，可重跑本脚本断点续传" | tee -a "$LOG"
    exit $RC2
fi

# ====== 重试轮：对异常日期最多再试 2 轮 ======
for round in 1 2; do
    [ -s "$FAILED" ] || break
    sort -u "$FAILED" > "$FAILED.tmp" && mv "$FAILED.tmp" "$FAILED"
    echo "[$(date '+%m-%d %H:%M:%S')] —— 重试轮 $round：$(wc -l < "$FAILED" | tr -d ' ') 个日期 ——"
    : > "$FAILED.round"
    while read -r d; do
        [ -n "$d" ] || continue
        window_check
        run_date "$d" || { echo "$d" >> "$FAILED.round"; [ $? -eq 3 ] && break; }
    done < "$FAILED"
    mv "$FAILED.round" "$FAILED"
done

if [ -s "$FAILED" ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] ⚠️ 仍有失败日期: $(tr '\n' ' ' < "$FAILED")（可重跑本脚本续传）" | tee -a "$LOG"
fi
echo "[$(date '+%m-%d %H:%M:%S')] ########## 全部补跑完成 ##########"
