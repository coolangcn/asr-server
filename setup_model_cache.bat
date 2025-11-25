@echo off
REM 设置模型缓存目录到D盘
REM 这个脚本会设置环境变量,让所有模型下载到D盘

echo ========================================
echo 配置模型缓存目录到 D:\AI\model_cache
echo ========================================

REM 创建模型缓存目录
if not exist "D:\AI\model_cache" mkdir "D:\AI\model_cache"
if not exist "D:\AI\model_cache\huggingface" mkdir "D:\AI\model_cache\huggingface"
if not exist "D:\AI\model_cache\modelscope" mkdir "D:\AI\model_cache\modelscope"
if not exist "D:\AI\model_cache\whisper" mkdir "D:\AI\model_cache\whisper"
if not exist "D:\AI\model_cache\torch" mkdir "D:\AI\model_cache\torch"

echo.
echo 已创建模型缓存目录:
echo   - D:\AI\model_cache\huggingface  (HuggingFace模型)
echo   - D:\AI\model_cache\modelscope   (ModelScope模型)
echo   - D:\AI\model_cache\whisper      (Whisper模型)
echo   - D:\AI\model_cache\torch        (PyTorch模型)
echo.

REM 设置环境变量
setx HF_HOME "D:\AI\model_cache\huggingface"
setx TRANSFORMERS_CACHE "D:\AI\model_cache\huggingface"
setx HUGGINGFACE_HUB_CACHE "D:\AI\model_cache\huggingface"
setx MODELSCOPE_CACHE "D:\AI\model_cache\modelscope"
setx XDG_CACHE_HOME "D:\AI\model_cache\whisper"
setx TORCH_HOME "D:\AI\model_cache\torch"

echo.
echo ========================================
echo 环境变量设置完成!
echo ========================================
echo.
echo 已设置以下环境变量:
echo   HF_HOME                = D:\AI\model_cache\huggingface
echo   TRANSFORMERS_CACHE     = D:\AI\model_cache\huggingface
echo   HUGGINGFACE_HUB_CACHE  = D:\AI\model_cache\huggingface
echo   MODELSCOPE_CACHE       = D:\AI\model_cache\modelscope
echo   XDG_CACHE_HOME         = D:\AI\model_cache\whisper
echo   TORCH_HOME             = D:\AI\model_cache\torch
echo.
echo ========================================
echo 重要提示:
echo ========================================
echo 1. 请 **重启命令行窗口** 或 **重启电脑** 使环境变量生效
echo 2. 如果C盘已有模型文件,可以手动移动到D盘:
echo    - C:\Users\Administrator\.cache\huggingface  -^> D:\AI\model_cache\huggingface
echo    - C:\Users\Administrator\.cache\modelscope   -^> D:\AI\model_cache\modelscope
echo    - C:\Users\Administrator\.cache\whisper      -^> D:\AI\model_cache\whisper
echo    - C:\Users\Administrator\.cache\torch        -^> D:\AI\model_cache\torch
echo 3. 移动后可以删除C盘的旧文件释放空间
echo.
echo 按任意键退出...
pause >nul
