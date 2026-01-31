@echo off
echo ============================================================
echo MUST++ Frontend Server
echo ============================================================
echo.
echo Starting lightweight API server on http://localhost:8080
echo.
echo Open your browser to: http://localhost:8080
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

cd /d "%~dp0.."
python api/app_lite.py
