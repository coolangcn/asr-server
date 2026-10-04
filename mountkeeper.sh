#!/bin/bash
# mountkeeper v2（root launchd 每5分钟执行，plist: /Library/LaunchDaemons/com.asr.mountkeeper.plist）
# 职责: ① 确保 /Volumes/download 挂载点目录存在（原有职责）
#       ② 路由劫持自愈（2026-09-22 事故: feth362 虚拟网卡把 192.168.1.0/24 劫持到
#          192.168.196.101，NAS/Sony 全断 4 小时）—— 用 /32 主机路由绕过，/32 永远优先于 /24
LOG=/Users/mac/asr-server/log/mountkeeper.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
# 心跳：每次运行摸一下时间戳，供远程确认守护进程存活（不会刷日志）
touch /var/tmp/mountkeeper.lastrun 2>/dev/null

# --- 职责①: 挂载点目录 ---
if [ ! -d /Volumes/download ]; then
    mkdir -p /Volumes/download && chown mac:staff /Volumes/download
    log "挂载点目录已重建"
fi

# --- 职责②: 路由自愈（仅当本机在 192.168.1.x 网段时）---
ifconfig en0 2>/dev/null | grep -q "inet 192.168.1." || exit 0

GW=$(route -n get 192.168.1.188 2>/dev/null | awk '/gateway:/{print $2}')
if [ "$GW" != "192.168.1.1" ]; then
    fixed=0
    for ip in 188 186 191; do
        route -n delete -host 192.168.1.$ip >/dev/null 2>&1
        if route -n add -host 192.168.1.$ip 192.168.1.1 >/dev/null 2>&1; then
            fixed=$((fixed+1))
        fi
    done
    log "检测到到NAS路由异常(gateway=${GW:-无})，已补 /32 主机路由 $fixed/3 条 (188=NAS 186=Pixel-6 191=Pixel-5)"
fi
