@echo off
title UPI Intelligence Architecture
cd /d "%~dp0"

echo.
echo  ================================================
echo   UPI Intelligence Architecture
echo  ================================================
echo.

REM ── Already running? ────────────────────────────────
netstat -an 2>nul | find ":8501" >nul 2>&1
if not errorlevel 1 (
    echo  Already running. Opening browser...
    timeout /t 1 /nobreak >nul
    start http://localhost:8501
    exit /b 0
)

REM ── Check Python ─────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install Python 3.9+ and re-run.
    pause
    exit /b 1
)

REM ── Install deps if missing ──────────────────────────
echo  [1/3] Checking dependencies...
python -c "import streamlit, langchain_groq, chromadb, langchain_huggingface, fitz" >nul 2>&1
if errorlevel 1 (
    echo        Installing — this takes a few minutes on first run...
    pip install -r requirements.txt --only-binary=:all: -q
    if errorlevel 1 (
        echo  ERROR: Dependency install failed.
        pause
        exit /b 1
    )
    echo        Done.
) else (
    echo        All dependencies present.
)

REM ── Run ingestion if vector store missing ────────────
if not exist "chroma_db\" (
    echo.
    echo  [2/3] First run: ingesting 127 circulars into vector store.
    echo        This takes 5-15 minutes. Please wait...
    echo.
    python ingest.py
    if errorlevel 1 (
        echo  ERROR: Ingestion failed.
        pause
        exit /b 1
    )
) else (
    echo  [2/3] Vector store found. Skipping ingestion.
)

REM ── Launch app detached from terminal ────────────────
echo.
echo  [3/3] Starting UPI Intelligence Architecture...
cscript //nologo "%~dp0launcher.vbs"

REM ── Wait for server to start then open browser ───────
echo        Waiting for server to start...
timeout /t 5 /nobreak >nul
start http://localhost:8501

echo.
echo  ================================================
echo   Running at: http://localhost:8501
echo   Logs saved: streamlit.log
echo   To stop:    run stop.bat
echo.
echo   You can close this window safely.
echo   The app will keep running in the background.
echo  ================================================
echo.
timeout /t 5 /nobreak >nul
