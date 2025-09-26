@echo off
echo Setting up PACA environment...
set PYTHONIOENCODING=utf-8
set PYTHONPATH=C:\Users\kk\claude\paca\paca_python
set PACA_LOG_LEVEL=INFO
chcp 65001 >nul 2>&1
echo Environment setup complete!
echo.
echo To run PACA:
echo python -m paca
pause
