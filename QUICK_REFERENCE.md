# 环境配置快速参考

## 🚀 快速开始

```bash
# 1. 复制模板
cp .env.example .env

# 2. 编辑配置
# 使用你喜欢的编辑器打开 .env 文件

# 3. 验证配置
./check_env.sh

# 4. 启动服务
./start_all.sh
```

## 📁 关键文件

| 文件 | 说明 | 是否提交 Git |
|------|------|-------------|
| `.env` | 实际配置（含敏感信息） | ❌ **绝不提交** |
| `.env.example` | 配置模板 | ✅ 可以提交 |
| `check_env.sh` | 验证脚本 | ✅ 可以提交 |

## 🔑 必填配置项

```bash
# Gemini API（智能摘要）
GEMINI_API_KEY=你的 API 密钥

# 邮箱通知（哭声警报）
EMAIL_AUTH_CODE=QQ 邮箱 SMTP 授权码
EMAIL_SENDER=你的 QQ 邮箱

# 服务配置
VENV_PATH=/Users/mac/asr_env  # macOS
# 或
VENV_PATH=D:\AI\asr_env       # Windows
```

## ✅ 验证清单

- [ ] `.env` 文件已创建
- [ ] 所有必填配置项已设置
- [ ] 运行 `./check_env.sh` 显示全部通过
- [ ] `.env` 文件不会被 Git 跟踪

## 🛡️ 安全提醒

- ⚠️ **不要**将 `.env` 文件提交到 Git
- ⚠️ **不要**在公开场合分享 `.env` 内容
- ✅ **定期**更新 API 密钥和授权码
- ✅ **备份** `.env` 文件到安全位置

## 🔧 故障排查

### 服务无法启动
```bash
# 检查 .env 文件
ls -la .env

# 验证配置
./check_env.sh
```

### 邮件发送失败
- 检查 QQ 邮箱是否开启 SMTP 服务
- 确认使用的是授权码而非密码
- 验证 SMTP 服务器和端口配置

### API 调用失败
- 验证 API 密钥是否有效
- 检查网络连接
- 确认 API 配额未超限

## 📚 详细文档

- `ENV_SETUP.md` - 完整配置指南
- `PRIVACY_SETUP_SUMMARY.md` - 实施总结

---

**提示**：更多详细信息请查看 `ENV_SETUP.md`
