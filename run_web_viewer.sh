#!/bin/bash
# Web Viewer (5009) 启动 wrapper —— 供 launchd 调用

cd /Users/mac/asr-server/nas-audio-notes-client || exit 1

set -a
source ../.env
set +a

exec /Users/mac/asr_env/bin/python3 web_viewer.py
