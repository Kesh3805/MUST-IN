@echo off
echo Starting MUST++ Lightweight Server...
echo.
cd /d "%~dp0.."
python api/app_lite.py
pause
