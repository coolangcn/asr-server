@echo off
chcp 65001 >nul
echo ======================================
echo   启动Web查看器 (Windows版)
echo ======================================
echo.
echo Web界面将在 http://localhost:5010 启动
echo.

python web_viewer.py

pause
