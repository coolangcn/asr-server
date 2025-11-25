@echo off
REM ========================================
REM 一键迁移模型到D盘并释放C盘空间
REM ========================================

echo.
echo ========================================
echo 模型缓存自动迁移工具
echo ========================================
echo.
echo 本脚本将:
echo   1. 创建D盘模型缓存目录
echo   2. 设置环境变量
echo   3. 移动C盘现有模型到D盘
echo   4. 删除C盘旧文件释放空间
echo.
echo 按任意键开始,或按 Ctrl+C 取消...
pause >nul

echo.
echo ========================================
echo [1/4] 创建D盘目录结构
echo ========================================
if not exist "D:\AI\model_cache" mkdir "D:\AI\model_cache"
if not exist "D:\AI\model_cache\huggingface" mkdir "D:\AI\model_cache\huggingface"
if not exist "D:\AI\model_cache\modelscope" mkdir "D:\AI\model_cache\modelscope"
if not exist "D:\AI\model_cache\whisper" mkdir "D:\AI\model_cache\whisper"
if not exist "D:\AI\model_cache\torch" mkdir "D:\AI\model_cache\torch"
echo ? 目录创建完成

echo.
echo ========================================
echo [2/4] 设置环境变量
echo ========================================
setx HF_HOME "D:\AI\model_cache\huggingface" >nul
setx TRANSFORMERS_CACHE "D:\AI\model_cache\huggingface" >nul
setx HUGGINGFACE_HUB_CACHE "D:\AI\model_cache\huggingface" >nul
setx MODELSCOPE_CACHE "D:\AI\model_cache\modelscope" >nul
setx XDG_CACHE_HOME "D:\AI\model_cache\whisper" >nul
setx TORCH_HOME "D:\AI\model_cache\torch" >nul
echo ? 环境变量设置完成

echo.
echo ========================================
echo [3/4] 移动模型文件到D盘
echo ========================================

REM 检查并移动 HuggingFace 模型
if exist "%USERPROFILE%\.cache\huggingface" (
    echo 正在移动 HuggingFace 模型...
    xcopy /E /I /Y /Q "%USERPROFILE%\.cache\huggingface" "D:\AI\model_cache\huggingface" >nul 2>&1
    if %errorlevel% == 0 (
        echo ? HuggingFace 模型移动完成
    ) else (
        echo ?? HuggingFace 模型移动失败
    )
) else (
    echo ?? 未找到 HuggingFace 模型
)

REM 检查并移动 ModelScope 模型
if exist "%USERPROFILE%\.cache\modelscope" (
    echo 正在移动 ModelScope 模型...
    xcopy /E /I /Y /Q "%USERPROFILE%\.cache\modelscope" "D:\AI\model_cache\modelscope" >nul 2>&1
    if %errorlevel% == 0 (
        echo ? ModelScope 模型移动完成
    ) else (
        echo ?? ModelScope 模型移动失败
    )
) else (
    echo ?? 未找到 ModelScope 模型
)

REM 检查并移动 Whisper 模型
if exist "%USERPROFILE%\.cache\whisper" (
    echo 正在移动 Whisper 模型...
    xcopy /E /I /Y /Q "%USERPROFILE%\.cache\whisper" "D:\AI\model_cache\whisper" >nul 2>&1
    if %errorlevel% == 0 (
        echo ? Whisper 模型移动完成
    ) else (
        echo ?? Whisper 模型移动失败
    )
) else (
    echo ?? 未找到 Whisper 模型
)

REM 检查并移动 Torch 模型
if exist "%USERPROFILE%\.cache\torch" (
    echo 正在移动 Torch 模型...
    xcopy /E /I /Y /Q "%USERPROFILE%\.cache\torch" "D:\AI\model_cache\torch" >nul 2>&1
    if %errorlevel% == 0 (
        echo ? Torch 模型移动完成
    ) else (
        echo ?? Torch 模型移动失败
    )
) else (
    echo ?? 未找到 Torch 模型
)

echo.
echo ========================================
echo [4/4] 删除C盘旧文件释放空间
echo ========================================
echo.
echo ?? 警告: 即将删除C盘旧模型文件!
echo.
echo 如果确认模型已成功移动到D盘,按任意键继续删除
echo 如果不确定,请按 Ctrl+C 取消,手动检查后再删除
pause >nul

REM 删除 HuggingFace
if exist "%USERPROFILE%\.cache\huggingface" (
    echo 正在删除 HuggingFace 旧文件...
    rmdir /S /Q "%USERPROFILE%\.cache\huggingface" 2>nul
    if not exist "%USERPROFILE%\.cache\huggingface" (
        echo ? HuggingFace 旧文件已删除
    ) else (
        echo ?? HuggingFace 旧文件删除失败
    )
)

REM 删除 ModelScope
if exist "%USERPROFILE%\.cache\modelscope" (
    echo 正在删除 ModelScope 旧文件...
    rmdir /S /Q "%USERPROFILE%\.cache\modelscope" 2>nul
    if not exist "%USERPROFILE%\.cache\modelscope" (
        echo ? ModelScope 旧文件已删除
    ) else (
        echo ?? ModelScope 旧文件删除失败
    )
)

REM 删除 Whisper
if exist "%USERPROFILE%\.cache\whisper" (
    echo 正在删除 Whisper 旧文件...
    rmdir /S /Q "%USERPROFILE%\.cache\whisper" 2>nul
    if not exist "%USERPROFILE%\.cache\whisper" (
        echo ? Whisper 旧文件已删除
    ) else (
        echo ?? Whisper 旧文件删除失败
    )
)

REM 删除 Torch
if exist "%USERPROFILE%\.cache\torch" (
    echo 正在删除 Torch 旧文件...
    rmdir /S /Q "%USERPROFILE%\.cache\torch" 2>nul
    if not exist "%USERPROFILE%\.cache\torch" (
        echo ? Torch 旧文件已删除
    ) else (
        echo ?? Torch 旧文件删除失败
    )
)

echo.
echo ========================================
echo 迁移完成!
echo ========================================
echo.
echo ? 所有操作已完成!
echo.
echo ?? 新的模型缓存位置:
echo    D:\AI\model_cache\huggingface
echo    D:\AI\model_cache\modelscope
echo    D:\AI\model_cache\whisper
echo    D:\AI\model_cache\torch
echo.
echo ?? 重要提示:
echo    1. 请 **重启命令行窗口** 使环境变量生效
echo    2. 重启后运行 start_all.bat 启动服务
echo    3. 如果模型加载失败,可能需要重新下载
echo.
echo 按任意键退出...
pause >nul
