@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ======================================
echo   启动 ASR 服务套件
echo ======================================
echo.
echo 正在启动...
echo - ASR API 服务 (端口 5008)
echo - Web 转录查看器 (端口 5009)
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"
call D:\AI\asr_env\Scripts\activate.bat
REM 进入 nas-audio-notes-client 目录启动 Web Viewer
cd nas-audio-notes-client
start "Web Viewer" D:\AI\asr_env\Scripts\python.exe web_viewer.py

REM 返回主目录
cd ..

REM 启动 ASR Server（包含文件监控、数据库保存、LLM 处理）
start "ASR Server" D:\AI\asr_env\Scripts\python.exe asr_server.py

REM 等待服务启动
timeout /t 5 /nobreak >nul

echo.
echo ======================================
echo   检查服务状态
echo ======================================
echo.

REM 检查 ASR Server (端口 5008)
netstat -ano | findstr ":5008.*LISTENING" >nul
if errorlevel 1 (
    echo    [警告] ASR服务(端口5008)未正常启动
) else (
    echo    ✓ ASR服务已启动 (端口5008)
)

REM 检查 Web Viewer (端口 5009)
netstat -ano | findstr ":5009.*LISTENING" >nul
if errorlevel 1 (
    echo    [警告] Web Viewer(端口5009)未正常启动
) else (
    echo    ✓ Web Viewer已启动 (端口5009)
)

echo.
echo ======================================
echo   所有服务启动完成！
echo ======================================
echo.
echo 📡 访问地址:
echo    - ASR服务管理: http://localhost:5008
echo    - 转录查看器:   http://localhost:5009
echo.
echo 📁 文件监控: 已集成到 ASR Server
echo 📊 数据库保存: 自动保存每次转录
echo 🤖 LLM 处理: 批量处理 (20条/批)
echo 📝 日志文件: log\asr-server.log
echo.
echo 提示: 关闭此窗口不会停止服务，服务在后台运行
echo.
pause
