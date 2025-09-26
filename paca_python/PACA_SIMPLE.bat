@echo off
REM PACA Simple Version - No Admin Required
title PACA Test

REM Go to script directory
cd /d "%~dp0"

REM Set encoding (works without admin)
chcp 65001 >nul 2>&1

cls
echo.
echo ================================================
echo            PACA Interactive Test
echo ================================================
echo.

REM Check Python
echo Checking Python...
python --version
if %errorLevel% neq 0 (
    echo.
    echo [ERROR] Python not found
    echo [HELP] Please install Python from https://python.org
    echo.
    pause
    exit /b 1
)

echo [OK] Python found

REM Check script file
if not exist "paca_ascii_only.py" (
    echo.
    echo [ERROR] paca_ascii_only.py not found
    echo [INFO] Current directory: %cd%
    echo [INFO] Available Python files:
    dir *.py /b
    echo.
    pause
    exit /b 1
)

echo [OK] Script file found

echo.
echo Starting PACA...
echo.
echo How to use:
echo   - Type: "python study" or "learn javascript"
echo   - Commands: "stats", "help", "quit"
echo   - Press Ctrl+C to force exit if needed
echo.

REM Run PACA with auto-close flag
python paca_ascii_only.py --auto-close

echo.
echo PACA session completed.
echo Press any key to close this window...
pause >nul