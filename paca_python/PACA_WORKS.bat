@echo off
REM PACA That Actually Works - No Unicode Issues
title PACA Test - Working Version

cd /d "%~dp0"

REM Admin check
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting admin privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d \"%~dp0\" && \"%~f0\"' -Verb RunAs"
    exit /b
)

cls
echo.
echo ================================================
echo            PACA Working Test Version
echo ================================================
echo.

REM Check Python
python --version
if %errorLevel% neq 0 (
    echo [ERROR] Python not found
    echo [HELP] Install from https://python.org
    pause
    exit /b 1
)

REM Check file
if not exist "paca_ascii_only.py" (
    echo [ERROR] paca_ascii_only.py not found
    echo [INFO] Current directory: %cd%
    dir *.py /b
    pause
    exit /b 1
)

echo [INFO] All checks passed
echo [INFO] Starting PACA...
echo.
echo Usage Examples:
echo   "python study"
echo   "learn javascript"
echo   "react help"
echo   "stats" for statistics
echo   "quit" to exit
echo.

REM Run the working version
python paca_ascii_only.py

echo.
echo [INFO] PACA session ended
pause