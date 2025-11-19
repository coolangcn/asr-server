@echo off
chcp 65001 >nul
title 音频转录系统 - 一键启动
color 0A

echo ========================================
echo    音频转录系统 - 一键启动
echo ========================================
echo.
echo [1/3] 正在启动 ASR 服务端...
echo.

:: 启动ASR服务端（在新窗口中）
cd /d "d:\AI\asr-server"
start "ASR服务端 (端口:5008)" cmd /k "d:\AI\asr_env\Scripts\activate && python asr_server.py"

:: 等待3秒让服务端启动
timeout /t 3 /nobreak >nul

echo [2/3] 正在启动转录客户端...
echo.

:: 启动转录客户端（在新窗口中）
cd /d "d:\AI\asr-server\nas-audio-notes-client"
start "转录客户端" cmd /k "python transcribe.py"

:: 等待2秒
timeout /t 2 /nobreak >nul

echo [3/3] 正在启动 Web 查看器...
echo.

:: 启动Web查看器（在新窗口中）
start "Web查看器 (端口:5010)" cmd /k "python web_viewer.py"

:: 等待2秒
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo    所有服务已启动完成！
echo ========================================
echo.
echo  ASR服务端:    http://localhost:5008
echo  Web查看器:    http://localhost:5010
echo.
echo  提示: 关闭此窗口不会停止服务
echo        请分别关闭各个服务窗口来停止服务
echo.
echo ========================================

:: 等待5秒后自动打开浏览器
timeout /t 3 /nobreak >nul
start http://localhost:5010

echo.
echo 按任意键关闭此窗口...
pause >nul
