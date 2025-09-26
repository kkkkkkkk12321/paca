@echo off
title PACA v5 - AI Assistant

echo PACA v5 시작 중...
cd /d "%~dp0"

python desktop_app/main.py

if errorlevel 1 (
    echo.
    echo 오류가 발생했습니다.
    echo Python이 설치되어 있는지 확인해주세요.
    pause
)