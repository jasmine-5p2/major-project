@echo off
echo ======================================================================
echo PhishGuard AI - Starting Backend Server
echo ======================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

REM Install requirements if needed
echo Checking dependencies...
pip install -r requirements.txt --quiet
echo.

REM Check if model files exist
if not exist "final_lstm_attention_phishing_detector.keras" (
    echo ERROR: Model file not found!
    echo Please place the following files in this directory:
    echo   - final_lstm_attention_phishing_detector.keras
    echo   - tokenizer.json
    echo   - label_classes.npy
    echo.
    pause
    exit /b 1
)

REM Start the server
echo Starting API server...
echo.
echo The API will be available at:
echo   - API: http://localhost:8000
echo   - Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py

pause