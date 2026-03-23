#!/bin/bash
# ==============================================
#   ASR Server 自动守护启动器 (macOS)
# ==============================================

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "======================================"
echo "  ASR Server 自动守护启动器 (macOS)"
echo "======================================"
echo ""

# 虚拟环境路径 (根据实际安装位置修改)
VENV_PATH="${VENV_PATH:-$HOME/asr_env}"

# 检查虚拟环境是否存在
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 虚拟环境不存在: $VENV_PATH"
    echo "请先创建虚拟环境，或设置 VENV_PATH 环境变量"
    echo "示例: export VENV_PATH=/path/to/your/venv"
    exit 1
fi

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 正在启动 ASR Server..."
    echo ""
    
    # 激活虚拟环境并启动服务
    source "$VENV_PATH/bin/activate"
    python3 asr_server.py
    
    EXIT_CODE=$?
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ ASR Server 已退出 (错误代码: $EXIT_CODE)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🕒 5秒后将进行自动重启..."
    echo ""
    
    sleep 5
done
