@echo off
setlocal

set "PROJECT_DIR=asr-server"
set "REPO_URL=https://github.com/coolangcn/asr-server.git"

if exist "%PROJECT_DIR%" (
    echo 项目目录 "%PROJECT_DIR%" 已存在，尝试拉取最新代码...
    cd "%PROJECT_DIR%"
    if %errorlevel% neq 0 (
        echo 无法进入目录 "%PROJECT_DIR%"，请手动检查。
        goto :eof
    )
    git pull
    if %errorlevel% equ 0 (
        echo 代码已更新到最新版本。
    ) else (
        echo 拉取代码失败，请检查网络或权限。
    )
) else (
    echo 项目目录 "%PROJECT_DIR%" 不存在，正在克隆仓库...
    git clone "%REPO_URL%"
    if %errorlevel% equ 0 (
        echo 仓库已成功克隆。
        cd "%PROJECT_DIR%"
        if %errorlevel% neq 0 (
            echo 无法进入目录 "%PROJECT_DIR%"，请手动检查。
            goto :eof
        )
    ) else (
        echo 克隆仓库失败，请检查网络或URL。
        goto :eof
    )
)

echo.
echo 当前工作目录: %cd%
echo.
echo 你可以继续执行安装依赖的步骤：
echo pip install -r requirements_stable.txt
echo.
pause
