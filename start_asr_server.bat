@echo off
title FunASR Server
cd /d "D:\AI\asr_server"
call "D:\AI\asr_env\Scripts\activate.bat"

echo Server starting... Logs are visible here AND in asr_server.log

rem 直接运行，不需要 > 符号
python asr_server.py

pause