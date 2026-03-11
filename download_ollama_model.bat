@echo off
chcp 65001 >nul
setlocal

set "MODEL_NAME=qwen3.5:4b"

echo ========================================
echo Download Required Ollama Model
echo ========================================
echo.
echo Required model: %MODEL_NAME%
echo This project is locked to the stable 4B path for quality and weak-device stability.
echo Make sure the model is installed before launching the packaged client.
echo.

ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama was not found. Install it before running this script.
    echo Download: https://ollama.com/download
    echo.
    pause
    exit /b 1
)

echo Downloading model: %MODEL_NAME%
echo This may take a few minutes. Keep your network connection stable.
echo.

ollama pull %MODEL_NAME%
if errorlevel 1 (
    echo.
    echo ========================================
    echo Download Failed
    echo ========================================
    echo.
    echo Check the following:
    echo 1. Network connectivity
    echo 2. Available disk space
    echo 3. Ollama service health
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Download Complete
echo ========================================
echo.
echo Model is ready: %MODEL_NAME%
echo You can now launch the packaged client. It will auto-check and start Ollama when needed.
echo.
pause
