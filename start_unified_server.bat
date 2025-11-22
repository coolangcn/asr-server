@echo off
REM 统一ASR服务启动脚本
REM 自动停止旧服务并启动新服务

echo ========================================
echo   统一ASR服务 - 启动脚本
echo ========================================
echo.

REM 激活Python虚拟环境
call D:\AI\asr_env\Scripts\activate.bat

REM 切换到项目目录
cd /d D:\AI\asr-server

REM ============ 停止旧的进程 ============
echo [1/4] 检查并停止旧进程...

REM 方法1: 通过窗口标题停止（适用于start命令启动的进程）
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *Unified ASR Server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *asr_server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *transcribe*" 2>nul

REM 方法2: 通过命令行参数停止（更可靠）
echo    正在查找并停止相关Python进程...
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr "PID:"') do (
    REM 检查该PID是否在运行我们的脚本
    wmic process where "ProcessId=%%i" get CommandLine 2>nul | findstr /I "unified_server\|asr_server\|transcribe.py" >nul
    if not errorlevel 1 (
        echo    停止进程 %%i
        taskkill /F /PID %%i 2>nul
    )
)

echo    等待进程完全结束...
timeout /t 3 >nul

REM 验证是否还有Python进程在使用端口5008
echo [2/4] 检查端口占用...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5008.*LISTENING"') do (
    echo    发现端口5008被进程 %%a 占用，正在停止...
    taskkill /F /PID %%a 2>nul
)

timeout /t 1 >nul

echo.
echo [3/4] 启动统一服务...
echo.

REM 启动unified_server.py（使用新的窗口标题）
start "Unified ASR Server" D:\AI\asr_env\Scripts\python.exe unified_server.py

REM 等待服务启动
echo    等待服务启动...
timeout /t 5 >nul

REM 验证服务是否成功启动
netstat -ano | findstr ":5008.*LISTENING" >nul
if errorlevel 1 (
    echo.
    echo [错误] 服务未能成功启动！请检查日志文件 asr-server.log
    echo.
    pause
    exit /b 1
)

echo.
echo [4/4] 服务启动完成！
echo.
echo 服务地址: http://localhost:5008
echo 日志文件: asr-server.log
echo.
echo 按任意键退出...
pause >nul
