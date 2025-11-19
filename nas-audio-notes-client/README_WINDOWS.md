# Windows本地部署指南

本文档说明如何在Windows系统上部署和运行音频转录客户端。

## 系统要求

- Windows 10/11
- Python 3.8+
- NAS网络驱动器映射到 `V:\`
- PostgreSQL数据库（运行在 192.168.1.188:5432）
- ffmpeg（需在系统PATH中）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：
- `flask` - Web框架
- `requests` - HTTP客户端
- `psycopg2-binary` - PostgreSQL数据库驱动

### 2. 配置检查

确保 `config.json` 配置正确：

```json
{
  "ASR_API_URL": "http://192.168.1.111:5008/transcribe",
  "SOURCE_DIR": "V:\\Sony-2",
  "TRANSCRIPT_DIR": "V:\\Sony-2\\transcripts",
  "PROCESSED_DIR": "V:\\Sony-2\\processed",
  "N8N_WEBHOOK_URL": "https://n8n.moco.fun/webhook/...",
  "DATABASE_URL": "postgresql://postgres:difyai123456@192.168.1.188:5432/postgres",
  "LOG_FILE_PATH": "transcribe.log",
  "WEB_PORT": 5010
}
```

**重要配置说明：**
- `SOURCE_DIR`: 音频文件监控目录（V盘映射的NAS路径）
- `DATABASE_URL`: PostgreSQL连接字符串
- `ASR_API_URL`: ASR服务器地址（需要先启动服务端）

### 3. 初始化数据库

首次运行前，测试数据库连接并初始化表结构：

```bash
python db_manager.py
```

成功输出示例：
```
测试数据库连接...
[DB] PostgreSQL连接成功: PostgreSQL 15.10 on x86_64-pc-linux-musl...
初始化连接池...
[DB] PostgreSQL连接池创建成功
初始化数据库表结构...
[DB] 数据库表结构初始化成功
```

### 4. 启动服务

#### 方式一：使用批处理脚本（推荐）

**启动转录服务：**
```bash
start_transcribe.bat
```

**启动Web查看器：**
```bash
start_web_viewer.bat
```

#### 方式二：手动启动

**转录服务：**
```bash
python transcribe.py
```

**Web查看器：**
```bash
python web_viewer.py
```

然后在浏览器访问：`http://localhost:5010`

## 工作流程

1. **音频采集**：将录音文件（.m4a, .mp3, .wav等）放入 `V:\Sony-2` 目录
2. **自动转录**：`transcribe.py` 监控该目录，自动调用ASR服务进行转录
3. **数据存储**：转录结果保存到PostgreSQL数据库和txt文件
4. **文件归档**：处理完的音频文件移动到 `V:\Sony-2\processed` 目录
5. **Web查看**：通过Web界面查看转录结果、时光对话和统计分析

## 目录结构

```
V:\Sony-2\
├── [音频文件].m4a          # 待处理的音频文件
├── transcripts\            # 转录文本文件
│   └── [文件名].txt
└── processed\              # 已处理的音频文件
    └── [音频文件].m4a
```

## 数据库架构

PostgreSQL数据库表结构：

```sql
CREATE TABLE transcriptions (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    full_text TEXT,
    segments_json TEXT
);

CREATE INDEX idx_created_at ON transcriptions(created_at DESC);
CREATE INDEX idx_filename ON transcriptions(filename);
```

## 故障排查

### 数据库连接失败

**错误信息：** `connection to server at "192.168.1.188", port 5432 failed`

**解决方案：**
1. 检查PostgreSQL服务是否运行
2. 检查网络连接：`ping 192.168.1.188`
3. 检查防火墙设置
4. 验证数据库凭据是否正确

### V盘无法访问

**错误信息：** `源目录不存在: V:\Sony-2`

**解决方案：**
1. 确认NAS网络驱动器已映射到V盘
2. 在资源管理器中打开 `V:\` 验证可访问性
3. 检查网络连接

### ffmpeg未找到

**错误信息：** `ffmpeg 转换失败`

**解决方案：**
1. 下载ffmpeg：https://ffmpeg.org/download.html
2. 将ffmpeg.exe添加到系统PATH
3. 或修改 `transcribe.py` 中的 `FFMPEG_PATH` 为完整路径

### ASR服务连接失败

**错误信息：** `无法连接服务端`

**解决方案：**
1. 确认ASR服务器（192.168.1.111:5008）正在运行
2. 检查网络连接
3. 验证 `config.json` 中的 `ASR_API_URL` 配置

## 日志查看

转录服务日志保存在 `transcribe.log`，可以通过以下方式查看：

```bash
# 查看最后20行
Get-Content transcribe.log -Tail 20

# 实时监控日志
Get-Content transcribe.log -Wait
```

## 配置管理

可以通过Web界面的"配置管理"页面在线修改配置，修改后会保存到 `config.json` 文件。

## 性能优化

- **数据库连接池**：使用连接池管理PostgreSQL连接，提高并发性能
- **索引优化**：在 `created_at` 和 `filename` 字段上创建索引，加速查询
- **异步处理**：转录服务采用轮询机制，每3秒检查一次新文件

## 安全建议

1. 修改PostgreSQL默认密码
2. 限制数据库访问IP范围
3. 使用HTTPS访问Web界面（需配置反向代理）
4. 定期备份数据库

## 技术支持

如遇问题，请检查：
1. `transcribe.log` - 转录服务日志
2. `web_viewer.log` - Web服务日志
3. PostgreSQL日志
