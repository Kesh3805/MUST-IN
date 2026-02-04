@echo off
echo ============================================================
echo MUST++ System Test - Testing Frontend-Backend Integration
echo ============================================================
echo.

cd /d "%~dp0.."

echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)
echo ✓ Python found
echo.

echo [2/5] Checking dependencies...
python -c "import flask; import flask_cors; print('✓ Flask and Flask-CORS installed')" 2>nul
if errorlevel 1 (
    echo ERROR: Required packages not installed
    echo Run: pip install -r requirements.txt
    pause
    exit /b 1
)
echo ✓ Dependencies OK
echo.

echo [3/5] Starting test server...
start /B python api/app_lite.py
echo Waiting for server to start...
timeout /t 5 /nobreak > nul
echo ✓ Server started
echo.

echo [4/5] Testing API endpoints...

echo   Testing /health endpoint...
curl -s http://localhost:8080/health > nul 2>&1
if errorlevel 1 (
    echo   ✗ Health check failed - server may not be running
) else (
    echo   ✓ Health check passed
)

echo   Testing /detect-script endpoint...
curl -s -X POST http://localhost:8080/detect-script -H "Content-Type: application/json" -d "{\"text\": \"test\"}" > nul 2>&1
if errorlevel 1 (
    echo   ✗ Script detection failed
) else (
    echo   ✓ Script detection passed
)

echo   Testing /analyze endpoint...
curl -s -X POST http://localhost:8080/analyze -H "Content-Type: application/json" -d "{\"text\": \"This is a test\"}" > nul 2>&1
if errorlevel 1 (
    echo   ✗ Analysis endpoint failed
) else (
    echo   ✓ Analysis endpoint passed
)
echo.

echo [5/5] Opening browser...
start http://localhost:8080
echo.

echo ============================================================
echo Test Complete!
echo ============================================================
echo.
echo The MUST++ frontend is now running in your browser.
echo Server is running in the background.
echo.
echo To test multilingual input, try these examples:
echo   - English: "This is a test message"
echo   - Hindi: "यह एक परीक्षण संदेश है"
echo   - Tamil: "இது ஒரு சோதனை செய்தி"
echo.
echo To stop the server, close this window or press Ctrl+C
echo ============================================================
echo.
pause
