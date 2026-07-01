@echo off
chcp 65001 >nul
title 清 (Qing) - AI Study Assistant

echo ============================================
echo   清 (Qing) — AI Study Assistant
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Opening Python download page...
    start https://www.python.org/downloads/
    echo Please install Python 3.10+ and run this script again.
    pause
    exit /b 1
)

:: Create venv if missing
if not exist "venv" (
    echo [1/3] First time setup — creating virtual environment...
    python -m venv venv
)

:: Activate
call venv\Scripts\activate

:: Install/update dependencies
echo [2/3] Installing dependencies...
pip install -r requirements.txt -q

:: Check .env
if not exist ".env" (
    copy .env.example .env >nul
    echo [INFO] Created .env from .env.example
)

:: Suppress harmless TF warnings
set TF_ENABLE_ONEDNN_OPTS=0

:: Start
echo [3/3] Starting 清 (Qing)...
echo.
echo    Opening http://localhost:7860
echo    Press Ctrl+C to stop
echo.
start http://localhost:7860
python -m app.main

pause
