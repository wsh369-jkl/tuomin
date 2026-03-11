@echo off
chcp 65001 >nul
echo ========================================
echo Ollama 安装指南
echo ========================================
echo.

echo 请按照以下步骤安装 Ollama:
echo.
echo 1. 访问 Ollama 官网
echo    https://ollama.com/download
echo.
echo 2. 下载 Windows 版本
echo    OllamaSetup.exe
echo.
echo 3. 运行安装程序
echo    双击安装，默认选项即可
echo.
echo 4. 验证安装
echo    打开命令行，输入: ollama --version
echo.
echo ========================================
echo.

echo 是否现在打开 Ollama 官网？(Y/N)
set /p choice=

if /i "%choice%"=="Y" (
    start https://ollama.com/download
)

echo.
echo 安装完成后，运行 download_ollama_model.bat 下载模型
echo.

pause
