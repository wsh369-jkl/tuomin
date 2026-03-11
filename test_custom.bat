@echo off
chcp 65001 >nul
echo ========================================
echo 测试自定义识别器
echo ========================================
echo.

cd backend
call venv\Scripts\activate.bat

python tests\test_custom.py

pause
