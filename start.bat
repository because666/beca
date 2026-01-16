@echo off
setlocal
title Quantitative Selection System Launcher

echo ==================================================
echo         Starting System...
echo ==================================================
echo.

REM 1. Check for Python command
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 'python' command not found.
    echo.
    echo Please check:
    echo 1. Is Python installed? (v3.8+ recommended)
    echo 2. Is "Add Python to PATH" checked during installation?
    echo.
    echo Trying to find 'py.exe' launcher...
    where py >nul 2>&1
    if %errorlevel% eq 0 (
        echo [INFO] Found py.exe, using it to start...
        py start.py
        goto :end
    )
    
    echo [FATAL] No Python interpreter found.
    echo Please install Python and try again.
    goto :end
)

REM 2. Run start script
echo [INFO] Python found, running start script...
python start.py

:end
echo.
echo ==================================================
echo         Program Exited
echo ==================================================
pause
endlocal
pause
