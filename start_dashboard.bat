@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: start_dashboard.bat - Background Streamlit Dashboard Launcher
:: =============================================================================
:: Requirements:
::   - Activates virtual environment (.venv / venv / PATH)
::   - Launches Streamlit dashboard in background without blocking
::   - Opens browser automatically to http://localhost:8501
::   - Avoids launching duplicate instances if port 8501 is already active
:: =============================================================================

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "LOGS_DIR=%PROJECT_DIR%logs"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

:: 1. Activate Python Environment
set "PYTHON_EXE="
if exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
    set "PYTHON_EXE=python"
) else if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
    set "PYTHON_EXE=python"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python environment not found for Streamlit dashboard.
        exit /b 1
    )
)

:: 2. Check if Dashboard is already listening on port 8501
powershell -NoProfile -Command "$client = New-Object System.Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', 8501); $true } catch { $false } finally { $client.Dispose() }" | findstr /i "True" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Streamlit dashboard is already running on http://localhost:8501
    start http://localhost:8501
    exit /b 0
)

:: 3. Launch Streamlit in background detached process
echo [INFO] Starting Streamlit dashboard on http://localhost:8501 ...
start "AI-Stock-Trader-Dashboard" /B %PYTHON_EXE% -m streamlit run "%PROJECT_DIR%dashboard\app.py" --server.port=8501 --server.headless=true --browser.gatherUsageStats=false > "%LOGS_DIR%\dashboard.log" 2>&1

:: 4. Brief delay to let server initialize, then open browser
timeout /t 3 /nobreak >nul
start http://localhost:8501

echo [SUCCESS] Dashboard launch initiated. Available at http://localhost:8501
exit /b 0
