@echo off
REM =====================================================================
REM  Personalised Content Recommendation - one-click launcher
REM  Runs the offline pipeline (when needed) then starts the REST API
REM  and the Streamlit dashboard.
REM
REM  Usage:
REM     run.bat            normal start (rebuilds only if artifacts are
REM                        missing or out of date)
REM     run.bat rebuild    force a full pipeline rebuild before starting
REM =====================================================================
setlocal
cd /d "%~dp0"

REM --- Locate a Python interpreter -------------------------------------
set "PY=C:\Users\RAHUL\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PY%" (
    where py >nul 2>nul && set "PY=py"
)
if "%PY%"=="" set "PY=python"
echo Using Python: %PY%

REM --- Install dependencies if key packages are missing -----------------
"%PY%" -c "import fastapi, streamlit, sklearn, textblob" 1>nul 2>nul
if errorlevel 1 (
    echo Installing dependencies from requirements.txt ...
    "%PY%" -m pip install -r requirements.txt
)

REM --- Decide whether the pipeline needs to run ------------------------
REM Rebuild if 'rebuild' was passed, or artifacts are missing, or the
REM processed dataset predates the url/description columns.
set "NEED_BUILD="
if /i "%1"=="rebuild" set "NEED_BUILD=1"

if not defined NEED_BUILD (
    "%PY%" -c "import os,sys,pandas as pd; ok=os.path.exists('data/embeddings.npy') and os.path.exists('data/processed_news.csv') and os.path.exists('data/user_simulation.csv') and ('url' in pd.read_csv('data/processed_news.csv', nrows=1).columns); sys.exit(0 if ok else 1)" 1>nul 2>nul
    if errorlevel 1 set "NEED_BUILD=1"
)

if defined NEED_BUILD (
    if /i "%1"=="rebuild" (
        echo Forcing full pipeline rebuild ...
        "%PY%" main.py --force
    ) else (
        echo Building / updating pipeline artifacts ...
        "%PY%" main.py
    )
) else (
    echo Pipeline artifacts are up to date - skipping build.
    echo   ^(use "run.bat rebuild" to regenerate everything^)
)

REM --- Start the REST API in its own window ----------------------------
echo.
echo Starting REST API   -^> http://127.0.0.1:8000/docs
start "Recommendation API" "%PY%" -m uvicorn app.api:app --host 127.0.0.1 --port 8000

REM --- Give the API a moment to load embeddings ------------------------
timeout /t 6 /nobreak >nul

REM --- Start the dashboard (this window) --------------------------------
echo Starting dashboard  -^> http://localhost:8501
echo.
"%PY%" -m streamlit run app/dashboard.py

endlocal
