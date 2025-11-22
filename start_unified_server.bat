@echo off
REM 统一ASR服务启动脚本

echo ========================================
echo   统一ASR服务 - 启动脚本
echo ========================================
echo.

REM 激活Python虚拟环境
call D:\AI\asr_env\Scripts\activate.bat

REM 切换到项目目录
cd /d D:\AI\asr-server

REM 停止旧的进程（可选）
echo [1/3] 检查并停止旧进程...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *unified_server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *asr_server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *transcribe*" 2>nul

timeout /t 2 >nul

echo.
echo [2/3] 启动统一服务...
echo.

REM 启动unified_server.py
start "Unified ASR Server" python unified_server.py

echo.
echo [3/3] 服务启动完成！
echo.
echo 服务地址: http://localhost:5008
echo 日志文件: asr-server.log
echo.
echo 按任意键退出...
pause >nul
