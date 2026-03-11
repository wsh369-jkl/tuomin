@echo off
chcp 65001 >nul
echo ========================================
echo 启动合同脱敏系统（Ollama版本）
echo ========================================
echo.

echo [1/2] 检查 Ollama 服务...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama 未安装或未启动
    echo.
    echo 请先确保:
    echo 1. Ollama 已安装
    echo 2. 模型已下载: ollama pull qwen3.5:4b
    echo.
    pause
    exit /b 1
)

echo ✅ Ollama 服务正常
echo.

echo [2/2] 启动 FastAPI 服务...
cd backend

echo 访问地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo.

REM 尝试使用 python，如果失败则使用 py
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python main.py
) else (
    py main.py
)

pause
