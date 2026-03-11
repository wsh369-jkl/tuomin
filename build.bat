@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%backend\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ========================================
echo Contract Desensitize - Build Script
echo ========================================
echo.

echo [1/3] Installing build dependencies...
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%build\requirements-build.txt"
if errorlevel 1 goto :error

echo.
echo [2/3] Building desktop client...
"%PYTHON_EXE%" "%SCRIPT_DIR%build\build.py"
if errorlevel 1 goto :error

echo.
echo [3/3] Generating Windows installer script...
"%PYTHON_EXE%" "%SCRIPT_DIR%build\package_windows_installer.py"
if errorlevel 1 goto :error

echo.
echo Build completed
echo Release directory: "%SCRIPT_DIR%release"
echo.
pause
exit /b 0

:error
echo.
echo Build failed. Review the output above.
echo.
pause
exit /b 1
