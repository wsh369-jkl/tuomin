@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%backend"
set "FRONTEND_DIR=%ROOT%frontend"
set "VENV_DIR=%BACKEND_DIR%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_BOOTSTRAP="

echo ========================================
echo Install Dependencies
echo ========================================
echo.

if not exist "%VENV_PYTHON%" (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_BOOTSTRAP=py -3"
        goto create_venv
    )

    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_BOOTSTRAP=python"
        goto create_venv
    )

    echo [ERROR] No Python runtime was found.
    echo [ERROR] Install Python or create backend\venv first.
    pause
    exit /b 1
)
goto install_backend

:create_venv
echo [INFO] Creating backend virtual environment...
%PYTHON_BOOTSTRAP% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create backend virtual environment.
    pause
    exit /b 1
)

:install_backend
echo [INFO] Installing backend dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m pip install -r "%BACKEND_DIR%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install backend dependencies.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found in PATH.
    echo [ERROR] Install Node.js first.
    pause
    exit /b 1
)

echo [INFO] Installing frontend dependencies...
cd /d "%FRONTEND_DIR%"
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies.
    pause
    exit /b 1
)

echo.
echo [INFO] Dependency installation completed.
echo [INFO] You can now run start_all.bat
echo.
pause
