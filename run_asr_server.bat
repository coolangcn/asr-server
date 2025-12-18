@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 切换到脚本所在目录
cd /d "%~dp0"

echo ======================================
20: echo   ASR Server 自动守护启动器
echo ======================================
echo.

:loop
echo [%date% %time%] 🚀 正在启动 ASR Server...
echo.

REM 激活虚拟环境并启动服
call D:\AI\asr_env\Scripts\activate.bat
D:\AI\asr_env\Scripts\python.exe asr_server.py

echo.
echo [%date% %time%] ⚠️ ASR Server 已退出 (错误代码: %errorlevel%)
echo [%date% %time%] 🕒 5秒后将进行自动重启...
echo.

timeout /t 5 /nobreak >nul
goto loop
