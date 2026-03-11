@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "FRONTEND_DIR=%ROOT%frontend"

echo ========================================
echo Start Frontend
echo ========================================
echo.

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found in PATH.
    echo [ERROR] Install Node.js first.
    pause
    exit /b 1
)

cd /d "%FRONTEND_DIR%"

if not exist "node_modules" (
    echo [INFO] Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies.
        pause
        exit /b 1
    )
)

echo [INFO] Frontend URL: http://localhost:5173
echo.

call npm run dev
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Frontend exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
