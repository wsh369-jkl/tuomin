@echo off
chcp 65001 >nul
echo ========================================
echo Compatibility Entry - 4B Model Download
echo ========================================
echo.
echo This project is locked to qwen3.5:4b. Redirecting to the required model download flow.
echo.
call "%~dp0download_ollama_model.bat"
