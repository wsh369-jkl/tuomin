@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%backend\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ========================================
echo Windows Local Platform-Independent Checks
echo ========================================
echo.
echo Coverage:
echo 1. qwen3.5:4b quality and recall regressions
echo 2. case-number digit-only masking regressions
echo 3. runtime security and stability regressions
echo 4. launcher, packaging, and installer logic regressions
echo.

echo [1/2] Running syntax checks...
"%PYTHON_EXE%" -m py_compile ^
  "%SCRIPT_DIR%backend\main.py" ^
  "%SCRIPT_DIR%backend\app\api\desensitize.py" ^
  "%SCRIPT_DIR%backend\app\core\runtime_probe.py" ^
  "%SCRIPT_DIR%desktop\main.py" ^
  "%SCRIPT_DIR%desktop\tray.py" ^
  "%SCRIPT_DIR%build\build.py" ^
  "%SCRIPT_DIR%build\package_windows_installer.py" ^
  "%SCRIPT_DIR%build\package_macos_installer.py"
if errorlevel 1 goto :error

echo.
echo [2/2] Running regression tests...
"%PYTHON_EXE%" -m pytest -q ^
  "%SCRIPT_DIR%backend\tests\test_runtime_hardening.py" ^
  "%SCRIPT_DIR%backend\tests\test_regression_guards.py" ^
  "%SCRIPT_DIR%backend\tests\test_llm_recall_regression.py" ^
  "%SCRIPT_DIR%backend\tests\test_appeal_regressions.py" ^
  "%SCRIPT_DIR%backend\tests\test_runtime_status.py" ^
  "%SCRIPT_DIR%backend\tests\test_desktop_launcher.py" ^
  "%SCRIPT_DIR%backend\tests\test_build_release.py" ^
  "%SCRIPT_DIR%backend\tests\test_installer_packaging.py"
if errorlevel 1 goto :error

echo.
echo ========================================
echo Windows local platform-independent checks passed
echo ========================================
echo.
call :maybe_pause %1
exit /b 0

:error
echo.
echo ========================================
echo Windows local platform-independent checks failed
echo ========================================
echo.
echo Review the output above and fix the failing item first.
echo.
call :maybe_pause %1
exit /b 1

:maybe_pause
if /I "%~1"=="--no-pause" exit /b 0
if "%NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0
