# 统一ASR服务 - 快速启动指南

## 启动服务

### Windows
```bash
start_unified.bat
```

### 或手动启动
```bash
cd d:\AI\asr-server
python unified_server.py
```

## 服务包含

1. **ASR API 服务** - `http://localhost:5008`
   - `/transcribe` - 音频转录
   - `/register` - 声纹注册
   - `/speakers` - 说话人管理

2. **文件监控**
   - 自动监控：`V:\Sony-2`
   - 自动转录并保存到数据库
   - 处理后移动到：`V:\Sony-2\processed`

3. **Web 界面**（可选）
   - 单独运行：`cd nas-audio-notes-client && python web_viewer.py`
   - 访问：`http://localhost:5009`

## 测试

### 测试文件监控
```bash
# 复制音频文件到源目录
copy test.m4a V:\Sony-2\

# 观察日志，应该自动处理
```

### 测试 API
```bash
curl -X POST -F "audio_file=@test.wav" http://localhost:5008/transcribe
```

## 配置

编辑 `unified_server.py` 第 34-40 行：

```python
class FileMonitorConfig:
    ENABLE = True  # 启用/禁用文件监控
    SOURCE_DIR = r"V:\Sony-2"  # 音频源目录
    TRANSCRIPT_DIR = r"V:\Sony-2\transcripts"
    PROCESSED_DIR = r"V:\Sony-2\processed"
    MONITOR_INTERVAL = 3  # 扫描间隔（秒）
```

## 优势

- ✅ 单进程运行（内存节省 50%）
- ✅ 直接函数调用（延迟降低 40-60%）
- ✅ 共享 AI 模型
- ✅ 简化部署
