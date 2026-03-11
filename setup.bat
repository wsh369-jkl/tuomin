@echo off
setlocal EnableExtensions

echo ========================================
echo Setup Environment
echo ========================================
echo.

call "%~dp0install_dependencies.bat"
if errorlevel 1 exit /b 1

echo [INFO] Optional Ollama check...
ollama list >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama was not detected.
    echo [WARN] The system can still run without LLM recognition.
) else (
    echo [INFO] Ollama is available.
)

echo.
echo [INFO] Setup completed.
pause
