@echo off
REM Batch file to activate virtual environment and launch Streamlit dashboard

echo ========================================
echo  Street Fighter Training Dashboard
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the venv folder exists in the current directory.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Check if app.py exists
if not exist "app.py" (
    echo ERROR: app.py not found!
    echo Please make sure you're running this from the correct directory.
    echo.
    pause
    exit /b 1
)

REM Check if database exists
if not exist "training_results.db" (
    echo WARNING: training_results.db not found!
    echo The dashboard may not work without a database.
    echo.
)

echo.
echo Starting Streamlit Dashboard...
echo The dashboard will open in your default web browser.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================
echo.

streamlit run app.py

REM If streamlit exits, pause to see any error messages
pause

