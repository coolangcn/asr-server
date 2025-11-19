@echo off
chcp 65001 >nul
echo ======================================
echo   启动音频转录服务 (Windows版)
echo ======================================
echo.

python transcribe.py

pause
