#!/bin/bash
# ASR Server (5008) 启动 wrapper —— 供 launchd 调用
# 职责：加载 .env 环境变量 + 激活 venv + 前台运行服务（崩溃由 launchd KeepAlive 拉起）

cd /Users/mac/asr-server || exit 1

set -a
source .env
set +a

exec /Users/mac/asr_env/bin/python3 asr_server.py
