@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%backend"
set "VENV_DIR=%BACKEND_DIR%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_BOOTSTRAP="

echo ========================================
echo Start Backend
echo ========================================
echo.

if exist "%VENV_PYTHON%" goto check_deps

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

:create_venv
echo [INFO] Creating backend virtual environment...
%PYTHON_BOOTSTRAP% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create backend virtual environment.
    pause
    exit /b 1
)

:check_deps
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Missing %VENV_PYTHON%
    pause
    exit /b 1
)

echo [INFO] Using %VENV_PYTHON%
echo [INFO] Checking backend dependencies...
"%VENV_PYTHON%" -c "import fastapi, pydantic, pydantic_settings, sqlalchemy" >nul 2>&1
if errorlevel 1 (
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
)

cd /d "%BACKEND_DIR%"
echo.
echo [INFO] Backend URL: http://localhost:8000
echo [INFO] API Docs:    http://localhost:8000/docs
echo.

"%VENV_PYTHON%" main.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Backend exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
