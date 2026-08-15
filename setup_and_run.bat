@echo off
setlocal
cd /d "%~dp0"

echo [WakeQuest] Preparing the local environment...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :error
)

echo [WakeQuest] Installing or updating dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [WakeQuest] Starting at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:error
echo.
echo Setup failed. Copy this window's error output and send it to Codex.
pause
exit /b 1

