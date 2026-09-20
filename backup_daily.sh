#!/bin/bash
# ==============================================
#   每日自动备份 (launchd 每天 03:30 调度)
#   备份内容：声纹库 JSON + 样本音频 + baby_cry_events 表
#   保留 14 天；顺带清理超大的 launchd 日志
# ==============================================
set -u

BASE=/Users/mac/asr-server
BACKUP_DIR=$BASE/backups
DATE=$(date +%Y-%m-%d)
DAY_DIR=$BACKUP_DIR/$DATE
LOG_FILE=$BACKUP_DIR/backup.log
PG_DUMP=/opt/homebrew/opt/libpq@18/bin/pg_dump
RETAIN_DAYS=14

mkdir -p "$DAY_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

FAIL=0

# 1) 声纹库 JSON
if [ -f "$BASE/speaker_db_multi.json" ]; then
    cp -f "$BASE/speaker_db_multi.json" "$DAY_DIR/" && log "OK  speaker_db_multi.json" || { FAIL=1; log "FAIL speaker_db_multi.json 复制失败"; }
else
    log "WARN 未找到 speaker_db_multi.json"
    FAIL=1
fi

# 2) 样本音频目录
if [ -d "$BASE/speaker_samples" ]; then
    tar -czf "$DAY_DIR/speaker_samples.tar.gz" -C "$BASE" speaker_samples 2>/dev/null \
        && log "OK  speaker_samples.tar.gz ($(du -h "$DAY_DIR/speaker_samples.tar.gz" | cut -f1))" \
        || { FAIL=1; log "FAIL speaker_samples 打包失败"; }
fi

# 3) Postgres baby_cry_events 表
if [ -x "$PG_DUMP" ]; then
    export PGPASSWORD=cncncncn
    if "$PG_DUMP" -h 192.168.1.188 -p 5433 -U postgres -d postgres -t baby_cry_events -Fc \
         -f "$DAY_DIR/baby_cry_events.dump" 2>>"$LOG_FILE"; then
        log "OK  baby_cry_events.dump ($(du -h "$DAY_DIR/baby_cry_events.dump" | cut -f1))"
    else
        FAIL=1; log "FAIL pg_dump 失败"
    fi
    unset PGPASSWORD
else
    log "FAIL 未找到 pg_dump ($PG_DUMP)"
    FAIL=1
fi

# 4) 清理 14 天前的旧备份
find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" -mtime +${RETAIN_DAYS} -exec rm -rf {} + 2>/dev/null

# 5) launchd 输出日志超过 50MB 时截断（app 自身日志已由 RotatingFileHandler 轮转）
for f in "$BASE"/log/launchd-*; do
    [ -f "$f" ] || continue
    SIZE=$(stat -f%z "$f" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 52428800 ]; then
        : > "$f"
        log "TRUNCATE $f (${SIZE} bytes)"
    fi
done

if [ "$FAIL" -eq 0 ]; then
    log "==== 备份完成: $DAY_DIR ===="
else
    log "==== 备份完成(有失败项): $DAY_DIR ===="
fi
