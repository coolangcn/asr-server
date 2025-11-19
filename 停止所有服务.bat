@echo off
chcp 65001 >nul
title 停止所有服务
color 0C

echo ========================================
echo    停止音频转录系统所有服务
echo ========================================
echo.

:: 关闭Python进程（asr_server.py, transcribe.py, web_viewer.py）
echo 正在停止服务...
echo.

taskkill /FI "WINDOWTITLE eq ASR服务端*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq 转录客户端*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Web查看器*" /F >nul 2>&1

echo 所有服务已停止！
echo.
timeout /t 2 /nobreak >nul
