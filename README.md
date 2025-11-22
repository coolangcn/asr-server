# AI录音存档系统 - 使用指南

## 快速开始

### 一键启动所有服务（推荐）
```batch
start_all.bat
```
这个命令会自动启动：
- ASR服务 + 文件监控 (端口5008)
- Web查看器 (端口5010)

## 访问方式

### 🎯 转录记录查看（主要使用）
- **本地访问**: http://localhost:5010
- **局域网访问**: http://192.168.1.111:5010

功能：
- **Dashboard**: 查看最新转录记录
- **Timeline**: 时间线对话视图
- **Statistics**: 说话人统计分析

### ⚙️ ASR服务管理
- **本地访问**: http://localhost:5008
- **局域网访问**: http://192.168.1.111:5008

功能：
- 说话人注册
- 说话人管理
- 实时日志查看

## 文件说明

### 启动脚本
- `start_all.bat` - **主启动脚本**（推荐使用）
  - 启动ASR服务 + 文件监控
  - 启动Web查看器
  - 自动清理旧进程

- `start_unified_server.bat` - 仅启动ASR服务
  - 只启动ASR服务和文件监控
  - 不启动Web查看器

- `停止所有服务.bat` - 停止所有Python服务

### 核心文件
- `unified_server.py` - 统一服务（ASR + 文件监控）
- `asr_server.py` - ASR核心服务
- `db_manager.py` - PostgreSQL数据库管理
- `speaker_db_multi.json` - 说话人声纹数据库

### 配置文件
- `requirements_stable.txt` - Python依赖包
- `GEMINI.md` - 项目说明文档

### 客户端文件（nas-audio-notes-client目录）
- `web_viewer.py` - Web查看器
- `transcribe.py` - 旧版客户端（已弃用，保留作为备份）

## 工作原理

### 架构
```
[音频文件 V:\Sony-2]
       ↓
[unified_server.py]
  ├── 文件监控线程（每3秒扫描）
  ├── ASR转录（直接函数调用，无HTTP）
  ├── 声纹识别
  └── 保存到PostgreSQL
       ↓
[web_viewer.py] ← 读取数据库
       ↓
[浏览器访问 :5010]
```

### 性能提升
- **旧架构**: 文件监控 --HTTP--> ASR服务
- **新架构**: 文件监控 --直接调用--> ASR服务
- **提升**: ~7% 更快，内存占用减半

## 配置说明

### 文件监控配置
编辑 `unified_server.py` 中的 `FileMonitorConfig` 类：

```python
class FileMonitorConfig:
    ENABLE = True  # 是否启用文件监控
    SOURCE_DIR = r"V:\Sony-2"  # 源音频目录
    TRANSCRIPT_DIR = r"V:\Sony-2\transcripts"  # TXT输出目录
    PROCESSED_DIR = r"V:\Sony-2\processed"  # 已处理文件目录
    MONITOR_INTERVAL = 3  # 扫描间隔（秒）
    SUPPORTED_EXTENSIONS = ('.m4a', '.acc', '.aac', '.mp3', '.wav', '.ogg', '.flac')
```

### 数据库配置
编辑 `db_manager.py` 中的 `DATABASE_URL`：

```python
DATABASE_URL = "postgresql://postgres:密码@IP:端口/数据库名"
```

## 故障排查

### 问题1: 端口被占用
**症状**: 启动失败，提示端口被占用

**解决**:
```batch
# 运行停止脚本
停止所有服务.bat
```

### 问题2: 数据库连接失败
**症状**: 日志显示 [DB Error]

**解决**: 检查数据库配置和网络连接
```batch
# 测试数据库连接
python db_manager.py
```

### 问题3: 模型加载失败
**症状**: 启动时卡在模型加载

**解决**:
1. 检查CUDA可用性: `nvidia-smi`
2. 确认modelscope缓存目录有足够空间
3. 首次运行需要下载模型，请耐心等待

## 日志文件

- `asr-server.log` - ASR服务和文件监控日志
- `nas-audio-notes-client/transcribe.log` - Web Viewer日志（如果独立运行）

## 注意事项

1. **虚拟环境**: 所有服务都在 `D:\AI\asr_env` 虚拟环境中运行
2. **文件监控**: 自动处理 `V:\Sony-2` 目录下的音频文件
3. **已处理文件**: 处理完的文件会移动到 `V:\Sony-2\processed`
4. **转录文本**: TXT格式的转录结果保存在 `V:\Sony-2\transcripts`

## 版本历史

### v2.0 (2025-11-22)
- ✅ 整合ASR服务和文件监控到单一进程
- ✅ 性能提升7%，内存占用减半
- ✅ 一键启动脚本
- ✅ 自动清理旧进程

### v1.0 (2025-11-19)
- 初始版本
- 独立的ASR服务和文件监控
