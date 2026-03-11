@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%backend"
set "DESKTOP_DIR=%ROOT%desktop"
set "VENV_DIR=%BACKEND_DIR%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

echo ========================================
echo Start Desktop
echo ========================================
echo.

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Missing backend virtual environment.
    echo [ERROR] Run install_dependencies.bat first.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import fastapi, pydantic, pydantic_settings, requests" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing backend dependencies...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    "%VENV_PYTHON%" -m pip install -r "%BACKEND_DIR%\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install backend dependencies.
        pause
        exit /b 1
    )
)

cd /d "%DESKTOP_DIR%"
"%VENV_PYTHON%" main.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Desktop launcher exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
