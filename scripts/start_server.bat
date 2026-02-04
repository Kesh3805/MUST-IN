@echo off
echo ============================================================
echo MUST++ Frontend Server
echo ============================================================
echo.
echo Choose server mode:
echo   [1] Lightweight - Quick startup, fallback-only
echo   [2] Full Pipeline - Includes transformer models
echo.
set /p MODE="Enter choice [1]: "
if "%MODE%"=="" set MODE=1
echo.

cd /d "%~dp0.."

if "%MODE%"=="2" goto FULLMODE
if "%MODE%"=="1" goto LITEMODE
goto LITEMODE

:FULLMODE
echo Starting FULL pipeline server with transformers...
echo This may take 30-60 seconds for model loading...
echo.
python api/app.py
goto END

:LITEMODE
echo Starting LIGHTWEIGHT server - fallback-only
echo Server will start in 2-3 seconds...
echo.
python api/app_lite.py
goto END

:END
echo.
echo Server is running on: http://localhost:8080
echo Open your browser to test the frontend
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
pause
