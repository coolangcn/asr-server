#!/bin/bash
# ==============================================
#   启动 ASR 服务套件 (macOS)
# ==============================================

cd "$(dirname "$0")"

echo ""
echo "======================================"
echo "  启动 ASR 服务套件 (macOS)"
echo "======================================"
echo ""

# 加载环境变量
if [ -f .env ]; then
    echo "📄 正在加载环境配置 (.env)..."
    export $(grep -v '^#' .env | xargs)
    echo "✅ 环境配置加载完成"
else
    echo "⚠️  未找到 .env 文件，使用默认配置"
fi
echo ""

echo "清理可能残留的旧进程..."
pkill -f "python3 asr_server.py" || true
pkill -f "python3 web_viewer.py" || true
sleep 1
echo ""
echo "正在启动..."
echo "- ASR API 服务 (端口 5008)"
echo "- Web 转录查看器 (端口 5009)"
echo ""

# 虚拟环境路径
VENV_PATH="${VENV_PATH:-$HOME/asr_env}"

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

# 启动 Web Viewer (后台运行)
echo "🌐 启动 Web Viewer..."
cd nas-audio-notes-client
python3 web_viewer.py &
WEB_PID=$!
cd ..

# 等待 Web Viewer 启动
sleep 2

# 启动 ASR Server (前台运行,自动重启)
echo "🚀 启动 ASR Server..."
while true; do
    python3 asr_server.py
    EXIT_CODE=$?
    echo ""
    echo "[$(date)] ⚠️ ASR Server 已退出 (错误代码: $EXIT_CODE)"
    echo "[$(date)] 🕒 5秒后将进行自动重启..."
    sleep 5
done
