@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: run_retrain.bat - Monthly Model Retraining Automated Execution Wrapper
:: =============================================================================
:: Requirements:
::   - Activates virtual environment (.venv or venv)
::   - Runs: python -m src.ml.retrain --force
::   - Logs stdout and stderr to logs/retrain_runner_YYYY-MM-DD.log
::   - Works regardless of invocation directory (uses %~dp0)
:: =============================================================================

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "LOGS_DIR=%PROJECT_DIR%logs"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

:: Determine current date in YYYY-MM-DD format using powershell
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%i"

set "TODAY_LOG=%LOGS_DIR%\retrain_runner_%TODAY%.log"

echo ============================================================================= >> "%TODAY_LOG%"
echo  AI STOCK TRADER - AUTOMATED MONTHLY RETRAIN RUN: %TODAY% >> "%TODAY_LOG%"
echo Started at: %TIME% on %DATE% >> "%TODAY_LOG%"
echo Project root: %PROJECT_DIR% >> "%TODAY_LOG%"
echo =============================================================================== >> "%TODAY_LOG%"

:: 1. Activate Python Virtual Environment
set "PYTHON_CMD="
if exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
    set "PYTHON_CMD=python"
    echo [INFO] Activated virtual environment from %PROJECT_DIR%.venv >> "%TODAY_LOG%"
) else if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
    set "PYTHON_CMD=python"
    echo [INFO] Activated virtual environment from %PROJECT_DIR%venv >> "%TODAY_LOG%"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
        echo [INFO] Virtual environment not found. Using system python from PATH. >> "%TODAY_LOG%"
    ) else (
        echo [ERROR] Python not found in PATH or virtual environment! >> "%TODAY_LOG%"
        exit /b 1
    )
)

:: 2. Execute Monthly Retraining Job
echo [INFO] Executing python -m src.ml.retrain --force >> "%TODAY_LOG%"
%PYTHON_CMD% -m src.ml.retrain --force >> "%TODAY_LOG%" 2>&1

set "RETRAIN_EXIT=!errorlevel!"
if !RETRAIN_EXIT! equ 0 (
    echo [SUCCESS] Monthly retraining job completed cleanly. >> "%TODAY_LOG%"
) else (
    echo [ERROR] Monthly retraining job failed with exit code !RETRAIN_EXIT!. >> "%TODAY_LOG%"
)

echo Finished at: %TIME% on %DATE% >> "%TODAY_LOG%"
exit /b !RETRAIN_EXIT!
