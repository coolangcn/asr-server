@echo off
REM 一键启动所有服务 - ASR服务 + Web Viewer

echo ========================================
echo   AI录音存档系统 - 一键启动
echo ========================================
echo.

REM 激活Python虚拟环境
call D:\AI\asr_env\Scripts\activate.bat

REM 切换到项目目录
cd /d D:\AI\asr-server

REM ============ 停止所有旧进程 ============
echo [1/5] 停止所有旧进程...

REM 方法1: 通过窗口标题停止
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *Unified ASR Server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *Web Viewer*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *asr_server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *transcribe*" 2>nul

REM 方法2: 通过命令行参数停止
echo    查找并停止相关Python进程...
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr "PID:"') do (
    wmic process where "ProcessId=%%i" get CommandLine 2>nul | findstr /I "unified_server\|web_viewer\|asr_server\|transcribe.py" >nul
    if not errorlevel 1 (
        echo    停止进程 %%i
        taskkill /F /PID %%i 2>nul
    )
)

echo    等待进程完全结束...
timeout /t 3 >nul

REM 检查端口占用
echo [2/5] 检查并释放端口...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5008.*LISTENING"') do (
    echo    释放端口5008 (PID %%a)
    taskkill /F /PID %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5009.*LISTENING"') do (
    echo    释放端口5009 (PID %%a)
    taskkill /F /PID %%a 2>nul
)

timeout /t 1 >nul

REM ============ 启动服务 ============
echo.
echo [3/5] 启动统一ASR服务...
start "Unified ASR Server" D:\AI\asr_env\Scripts\python.exe unified_server.py

echo    等待ASR服务启动...
timeout /t 5 >nul

echo.
echo [4/5] 启动Web Viewer...
cd /d D:\AI\asr-server\nas-audio-notes-client
start "Web Viewer" D:\AI\asr_env\Scripts\python.exe web_viewer.py --source-path V:\Sony-2

echo    等待Web Viewer启动...
timeout /t 3 >nul

REM ============ 验证服务 ============
echo.
echo [5/5] 验证服务状态...

netstat -ano | findstr ":5008.*LISTENING" >nul
if errorlevel 1 (
    echo    [警告] ASR服务(端口5008)未正常启动
) else (
    echo    ? ASR服务已启动 (端口5008)
)

netstat -ano | findstr ":5009.*LISTENING" >nul
if errorlevel 1 (
    echo    [警告] Web Viewer(端口5009)未正常启动
) else (
    echo    ? Web Viewer已启动 (端口5009)
)

echo.
echo ========================================
echo   所有服务启动完成！
echo ========================================
echo.
echo ? 访问地址:
echo    - ASR服务管理: http://localhost:5008
echo    - 转录查看器:   http://localhost:5009
echo.
echo ? 文件监控:     V:\Sony-2
echo ? 日志文件:     asr-server.log
echo.
echo 提示: 关闭此窗口不会停止服务，服务在后台运行
echo.
pause
