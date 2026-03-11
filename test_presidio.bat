@echo off
chcp 65001 >nul
echo ========================================
echo 测试 Presidio 规则引擎
echo ========================================
echo.

cd backend
call venv\Scripts\activate.bat

python tests\test_presidio.py

pause
