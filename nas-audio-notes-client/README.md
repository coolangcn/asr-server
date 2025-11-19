# AI 录音存档系统部署指南

## 系统概述
AI录音存档系统是一个用于实时转录、存储和查看录音文件的自动化系统，包含两个核心组件：

- **web_viewer.py**: Web界面查看器，提供仪表盘和时光对话两种视图
- **transcribe.py**: 实时录音监控器，自动处理录音文件并上传到转录服务器

## 部署方式

### 1. Linux 系统一键部署

#### 使用说明
```bash
# 下载脚本
git clone <repository-url>
cd nas

# 赋予执行权限
chmod +x one_click_deploy.sh stop_services.sh restart_services.sh

# 执行一键部署（需root权限）
sudo ./one_click_deploy.sh
```

#### 功能说明
- 自动安装Python环境和依赖包
- 创建并配置虚拟环境
- 初始化SQLite数据库
- 自动启动Web服务和转录服务
- 创建停止/重启服务脚本

### 2. Windows 系统一键部署

#### 使用说明
```powershell
# 下载脚本到d:\AI\nas目录
cd d:\AI\nas

# 以管理员身份运行PowerShell
# 执行一键部署
.\one_click_deploy.ps1
```

#### 功能说明
- 自动检测并安装Python 3.11
- 安装所需依赖包（flask, requests）
- 创建配置目录和数据库
- 启动Web服务和转录服务

### 3. Docker 部署（可选）

#### Dockerfile
```dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5009

CMD ["python", "web_viewer.py"]
```

#### 构建和运行
```bash
# 构建镜像
docker build -t ai-nas-web .

# 运行容器
docker run -d -p 5009:5009 --name ai-nas-web ai-nas-web
```

## 服务管理

### Linux 系统
```bash
# 停止服务
./stop_services.sh

# 重启服务
./restart_services.sh

# 查看日志
tail -f web_viewer.log transcribe.log
```

### Windows 系统
```powershell
# 手动停止服务
Get-Process -Name python | Where-Object {$_.CommandLine -like "*web_viewer.py*"} | Stop-Process
Get-Process -Name python | Where-Object {$_.CommandLine -like "*transcribe.py*"} | Stop-Process

# 手动启动服务
Start-Process python web_viewer.py
Start-Process python transcribe.py
```

## 配置说明

### 核心配置文件
- **web_viewer.py**: Web服务配置
  - `DB_PATH`: 数据库路径
  - `SOURCE_DIR`: 录音文件目录
  - `WEB_PORT`: Web服务端口

- **transcribe.py**: 转录服务配置
  - `ASR_HTTP_URL`: 转录API地址
  - `SOURCE_DIR`: 录音源目录
  - `TRANSCRIPT_DIR`: 转录结果目录
  - `DB_PATH`: 数据库路径

## 访问方式

### Web 界面
```
地址: http://[服务器IP]:5009
视图: 
1. 仪表盘 - 显示系统状态和所有转录记录
2. 时光对话 - 以聊天方式展示转录内容
```

### API 接口
```
GET /api/status - 获取系统状态
GET /api/data - 获取所有转录记录
```

## 技术支持

### 常见问题
1. **端口被占用**
   - 修改`WEB_PORT`配置
   - 执行`lsof -i :5009`（Linux）或`netstat -ano | findstr :5009`（Windows）查找占用进程

2. **转录失败**
   - 检查转录服务器是否在线
   - 查看`transcribe.log`日志

3. **数据库问题**
   - 检查数据库文件权限
   - 重新初始化数据库：`rm /volume2/download/records/Sony-2/transcripts.db && python -c "from transcribe import init_db; init_db()"`

## 更新记录

- v1.0.0 (2025-01-01): 初始版本
- v1.0.1 (2025-01-02): 修复空白文本显示问题
- v1.0.2 (2025-01-03): 增加一键部署脚本支持

## 许可证

MIT License