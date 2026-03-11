@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"

echo ========================================
echo Start Full System
echo ========================================
echo.

echo [INFO] Checking Ollama service...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ok = $false; try { $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:11434/api/tags' -TimeoutSec 2; if ($resp.StatusCode -eq 200) { $ok = $true } } catch {}; if (-not $ok) { $cmd = Get-Command ollama -ErrorAction SilentlyContinue; if ($cmd) { Start-Process -FilePath $cmd.Source -ArgumentList 'serve'; Start-Sleep -Seconds 3; try { $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5; if ($resp.StatusCode -eq 200) { exit 0 } } catch {}; exit 2 } else { exit 3 } }"
set "OLLAMA_STATUS=%ERRORLEVEL%"
if "%OLLAMA_STATUS%"=="2" (
    echo [WARN] Ollama was started but is not ready yet. LLM recognition may still need a few more seconds.
)
if "%OLLAMA_STATUS%"=="3" (
    echo [WARN] Ollama is not installed or not in PATH. The app can still start, but only regex/custom recognition will work.
)
echo.

echo [INFO] Starting backend...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%ROOT%start_backend.bat' -WorkingDirectory '%ROOT%'"
if errorlevel 1 (
    echo [ERROR] Failed to start backend window.
    pause
    exit /b 1
)

timeout /t 2 /nobreak >nul

echo [INFO] Starting frontend...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%ROOT%start_frontend.bat' -WorkingDirectory '%ROOT%'"
if errorlevel 1 (
    echo [ERROR] Failed to start frontend window.
    pause
    exit /b 1
)

echo.
echo [INFO] Frontend: http://localhost:5173
echo [INFO] Backend:  http://localhost:8000
echo [INFO] Docs:     http://localhost:8000/docs
echo.
pause
