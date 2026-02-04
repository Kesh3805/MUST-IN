@echo off
REM Quick Start Script for MUST-IN Transformer Training
REM Downloads models and starts training

echo.
echo ========================================
echo MUST-IN Quick Start - Transformer Training
echo ========================================
echo.

echo [1/4] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] Installing dependencies...
pip install -q torch transformers scikit-learn pandas numpy
echo Done.
echo.

echo [3/4] Downloading transformer models...
echo.
echo Available options:
echo   1. Quick Start (mBERT only, ~400MB)
echo   2. Full Setup (All models, ~1.2GB)
echo   3. Skip download (models already downloaded)
echo.
set /p choice="Select option (1-3): "

if "%choice%"=="1" (
    echo.
    echo Downloading mBERT cased model...
    python scripts/download_transformers.py --model mbert-cased
) else if "%choice%"=="2" (
    echo.
    echo Downloading all models...
    python scripts/download_transformers.py --all
) else (
    echo.
    echo Skipping download...
)
echo.

echo [4/4] Verifying models...
python scripts/download_transformers.py --verify
echo.

echo ========================================
echo Ready to train!
echo ========================================
echo.
echo Choose training mode:
echo   1. Quick Test (1 epoch, ~30 mins)
echo   2. Standard Training (3 epochs, ~2-3 hours)
echo   3. Full Benchmark (All models, ~6-8 hours)
echo   4. Skip training (manual later)
echo.
set /p train_choice="Select option (1-4): "

if "%train_choice%"=="1" (
    echo.
    echo Starting quick test training...
    python main.py --run-dl
) else if "%train_choice%"=="2" (
    echo.
    echo Starting standard training...
    python main.py --run-dl --save-models --generate-report
) else if "%train_choice%"=="3" (
    echo.
    echo Starting full benchmark...
    python main.py --run-dl --run-xlm --save-models --generate-report
) else (
    echo.
    echo Training skipped. Run manually:
    echo   python main.py --run-dl --save-models
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   - View results: start results\results_summary.html
echo   - Start API: python api\app.py
echo   - See documentation: TRAINING_WORKFLOW.md
echo.
pause
