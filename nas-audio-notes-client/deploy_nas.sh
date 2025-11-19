#!/bin/bash

# ================= 配置区域 =================
PYTHON_PATH="/bin/python3"
SCRIPT_DIR="/volume1/docker/scripts/nas_ai_audio_notes"

# 需要运行的两个核心脚本
WEB_SCRIPT="web_viewer.py"
MONITOR_SCRIPT="transcribe.py"

# 源路径配置 (仅适用于Linux系统)
# 可以通过环境变量 SOURCE_PATH_OVERRIDE 来覆盖默认路径
DEFAULT_SOURCE_PATH="/volume2/download/records/Sony-2"
SOURCE_PATH="${SOURCE_PATH_OVERRIDE:-$DEFAULT_SOURCE_PATH}"

# 日志路径配置
LOG_FILE_PATH="${SCRIPT_DIR}/transcribe.log"
# ===========================================

echo "=== 开始执行 NAS 部署 (幂等模式) ==="
echo "时间: $(date)"
echo "源路径: $SOURCE_PATH"
echo "如需更改源路径，请设置环境变量 SOURCE_PATH_OVERRIDE"
echo "例如: export SOURCE_PATH_OVERRIDE=/path/to/your/audio/files && ./deploy_nas.sh"

# --- 核心函数: 确保服务运行 (幂等) ---
ensure_service_running() {
    local SCRIPT_NAME=$1
    local FULL_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
    local LOG_FILE="${SCRIPT_DIR}/${SCRIPT_NAME%.*}.log"

    echo "正在检查: $SCRIPT_NAME ..."

    # 1. 获取正在运行的进程数量
    # 使用 grep 技巧 [f]ile.py 避免匹配到 grep 命令本身
    # wc -l 用于统计行数
    local PROC_COUNT=$(ps aux | grep "[${SCRIPT_NAME:0:1}]${SCRIPT_NAME:1}" | wc -l)

    # 如果有进程在运行，杀掉它们
    if [ "$PROC_COUNT" -ge 1 ]; then
        local PIDS=$(ps aux | grep "[${SCRIPT_NAME:0:1}]${SCRIPT_NAME:1}" | awk '{print $2}')
        echo "  [停止] 检测到 $PROC_COUNT 个 $SCRIPT_NAME 实例在运行 (PID: $PIDS)。正在停止..."
        pkill -f "$SCRIPT_NAME"
        # 等待进程完全停止
        sleep 1
        # 再次检查是否还有进程残留
        local REMAINING=$(ps aux | grep "[${SCRIPT_NAME:0:1}]${SCRIPT_NAME:1}" | wc -l)
        if [ "$REMAINING" -gt 0 ]; then
            echo "  [强制停止] 仍有 $REMAINING 个实例残留，正在强制停止..."
            pkill -9 -f "$SCRIPT_NAME"
            sleep 1
        fi
    fi

    # 启动新的实例
    echo "  [启动] 正在启动 $SCRIPT_NAME..."
    # 对于transcribe.py和web_viewer.py，传递源路径参数
    if [ "$SCRIPT_NAME" = "$MONITOR_SCRIPT" ] || [ "$SCRIPT_NAME" = "$WEB_SCRIPT" ]; then
        nohup $PYTHON_PATH "$FULL_PATH" --source-path "$SOURCE_PATH" > "$LOG_FILE" 2>&1 &
    else
        nohup $PYTHON_PATH "$FULL_PATH" > "$LOG_FILE" 2>&1 &
    fi
    sleep 2 # 等待进程初始化

    # 二次确认启动结果
    if ps aux | grep -q "[${SCRIPT_NAME:0:1}]${SCRIPT_NAME:1}"; then
        local NEW_PID=$(ps aux | grep "[${SCRIPT_NAME:0:1}]${SCRIPT_NAME:1}" | awk '{print $2}')
        echo "  [成功] $SCRIPT_NAME 已成功启动 (PID: $NEW_PID)。"
    else
        echo "  [失败] $SCRIPT_NAME 启动失败！请查看日志: $LOG_FILE"
    fi
    echo "-----------------------------------"
}

# --- 执行部署逻辑 ---

# 1. 检查并部署 Web 查看器
ensure_service_running "$WEB_SCRIPT"

# 2. 检查并部署 实时监控器
ensure_service_running "$MONITOR_SCRIPT"

echo "=== 部署检查完成 ==="
echo "Web 界面地址: http://[NAS_IP]:5009"
echo "监控模式: 实时监控中"