@echo off
chcp 65001 >nul
title 安装依赖包

echo ========================================
echo 安装系统依赖包
echo ========================================
echo.

echo 正在安装依赖包，这可能需要几分钟时间...
echo.

pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo 依赖包安装成功！
    echo ========================================
    echo.
    echo 现在可以运行"启动系统.bat"来启动系统
) else (
    echo.
    echo ========================================
    echo 依赖包安装失败！
    echo ========================================
    echo.
    echo 请检查网络连接或尝试使用国内镜像源：
    echo pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
pause
