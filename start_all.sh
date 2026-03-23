#!/bin/bash
cd "$(dirname "$0")"
source ~/asr_env/bin/activate

# 杀掉占用 5009 端口的进程
lsof -ti:5009 | xargs kill -9
# 杀掉占用 5008 端口的进程（如果有）
lsof -ti:5008 | xargs kill -9

echo "🌐 启动 Web Viewer (端口 5009)..."
cd nas-audio-notes-client
python3 web_viewer.py &
cd ..

echo "🚀 启动 ASR Server (端口 5008)..."
python3 asr_server.py
