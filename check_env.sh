#!/bin/bash
# ==============================================
#   验证环境配置
# ==============================================

cd "$(dirname "$0")"

echo ""
echo "======================================"
echo "  环境配置验证"
echo "======================================"
echo ""

# 检查 .env 文件
if [ -f .env ]; then
    echo "✅ .env 文件存在"
else
    echo "❌ .env 文件不存在"
    echo "   提示：请复制 .env.example 为 .env 并配置"
    exit 1
fi

# 检查 .gitignore
if grep -q "^\.env$" .gitignore; then
    echo "✅ .gitignore 已正确配置忽略 .env 文件"
else
    echo "❌ .gitignore 未配置忽略 .env 文件"
fi

# 检查关键配置项
echo ""
echo "检查配置项:"

check_env_var() {
    local var_name=$1
    local var_value=$(grep "^${var_name}=" .env | cut -d'=' -f2)
    if [ -n "$var_value" ]; then
        echo "  ✅ $var_name 已配置"
    else
        echo "  ⚠️  $var_name 未配置"
    fi
}

check_env_var "GEMINI_API_KEY"
check_env_var "EMAIL_AUTH_CODE"
check_env_var "EMAIL_SENDER"
check_env_var "VENV_PATH"

echo ""
echo "======================================"
echo "  配置验证完成"
echo "======================================"
echo ""

# 显示 Git 状态（确认 .env 不会被提交）
echo "Git 状态检查:"
git status --porcelain .env 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ .env 文件不会被 Git 跟踪"
else
    echo "ℹ️  当前目录不是 Git 仓库或 Git 未安装"
fi

echo ""
