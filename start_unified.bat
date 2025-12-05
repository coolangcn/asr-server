@echo off
chcp 65001 >nul
echo ======================================
echo   启动统一ASR服务
echo ======================================
echo.
echo 正在启动...
echo - ASR API 服务 (端口 5008)
echo - 文件监控和自动转录
echo - 数据库保存
echo.

cd /d d:\AI\asr-server
python unified_server.py

pause
