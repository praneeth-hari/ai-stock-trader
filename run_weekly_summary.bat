@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "LOGS_DIR=%PROJECT_DIR%logs"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

set "PYTHON_CMD="
if exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
    set "PYTHON_CMD=python"
) else if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
    set "PYTHON_CMD=python"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
    ) else (
        echo [ERROR] Python environment not found for weekly summary report.
        exit /b 1
    )
)

%PYTHON_CMD% "%PROJECT_DIR%src\reports\weekly_summary.py"
exit /b %ERRORLEVEL%
