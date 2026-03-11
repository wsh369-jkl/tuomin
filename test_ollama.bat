@echo off
chcp 65001 >nul
echo ========================================
echo 测试 Ollama 服务
echo ========================================
echo.

cd /d "%~dp0backend"

"C:\Users\29376\AppData\Local\Python\bin\python.exe" tests\test_ollama.py

if %errorlevel% neq 0 (
    echo.
    echo 测试失败！可能的原因：
    echo 1. 缺少依赖，请运行: pip install requests
    echo 2. Ollama 服务未启动
    echo 3. 模型未下载: ollama pull qwen3.5:4b
)

pause
