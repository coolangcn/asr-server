# 环境配置说明

## 概述

本项目使用 `.env` 文件来管理敏感配置信息（如 API 密钥、邮箱凭证等），确保这些隐私信息不会被提交到 Git 版本控制系统。

## 文件说明

- **`.env`** - 实际的环境配置文件（包含真实敏感信息，**不会被 Git 提交**）
- **`.env.example`** - 环境配置示例模板（可以安全提交到 Git）
- **`.gitignore`** - Git 忽略规则，已配置忽略所有 `.env` 文件

## 快速开始

### 1. 创建环境配置文件

复制示例模板创建你的 `.env` 文件：

```bash
cp .env.example .env
```

### 2. 编辑配置文件

打开 `.env` 文件，填入你的真实配置信息：

```bash
# Gemini LLM 配置
GEMINI_API_KEY=你的 Gemini API 密钥
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_MODEL_NAME=gemini-2.5-pro

# 邮箱配置
EMAIL_SMTP_SERVER=smtp.qq.com
EMAIL_SMTP_PORT=465
EMAIL_SENDER=你的 QQ 邮箱
EMAIL_AUTH_CODE=你的 QQ 邮箱 SMTP 授权码
EMAIL_RECEIVER=接收通知的邮箱地址

# 服务配置
ASR_SERVER_HOST=0.0.0.0
ASR_SERVER_PORT=5008
WEB_VIEWER_PORT=5009

# 虚拟环境路径
VENV_PATH=/Users/mac/asr_env  # macOS
# 或
VENV_PATH=D:\AI\asr_env  # Windows
```

### 3. 启动服务

启动脚本会自动加载 `.env` 文件中的配置：

```bash
# macOS/Linux
./start_all.sh

# Windows
start_all.bat
```

## 配置项说明

### Gemini LLM 配置

- `GEMINI_API_KEY` - Google Gemini API 密钥
- `GEMINI_API_BASE_URL` - Gemini API 基础 URL
- `GEMINI_MODEL_NAME` - 使用的 Gemini 模型名称

### 邮箱配置

- `EMAIL_SMTP_SERVER` - SMTP 服务器地址（如 QQ 邮箱：smtp.qq.com）
- `EMAIL_SMTP_PORT` - SMTP 端口（SSL：465）
- `EMAIL_SENDER` - 发件人邮箱
- `EMAIL_AUTH_CODE` - 邮箱 SMTP 授权码（不是密码）
- `EMAIL_RECEIVER` - 收件人邮箱

### 服务配置

- `ASR_SERVER_HOST` - ASR 服务监听地址
- `ASR_SERVER_PORT` - ASR 服务端口
- `WEB_VIEWER_PORT` - Web 查看器端口

### 虚拟环境路径

- `VENV_PATH` - Python 虚拟环境路径

## 安全提示

1. **不要将 `.env` 文件提交到 Git** - `.gitignore` 已配置自动忽略
2. **获取邮箱授权码** - QQ 邮箱需要在设置中开启 SMTP 服务并生成授权码
3. **保护 API 密钥** - Gemini API 密钥有使用限额，请妥善保管
4. **备份配置** - 建议将 `.env` 文件备份到安全位置

## 故障排查

### 服务无法读取环境变量

确保启动脚本正确加载了 `.env` 文件：

```bash
# 检查 .env 文件是否存在
ls -la .env

# 手动测试环境变量是否生效
echo $GEMINI_API_KEY
```

### 邮件发送失败

1. 检查 SMTP 授权码是否正确
2. 确认邮箱已开启 SMTP 服务
3. 检查防火墙是否阻止了 SMTP 端口

### API 调用失败

1. 验证 API 密钥是否有效
2. 检查网络连接
3. 确认 API 配额未超限

## 开发建议

- 在开发环境中使用测试 API 密钥
- 生产环境使用独立的生产配置
- 定期更新 API 密钥和授权码
- 使用环境变量管理工具（如 direnv）提高开发效率
