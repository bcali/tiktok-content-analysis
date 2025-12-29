@echo off
setlocal

REM Prefer the Python launcher 'py', else fall back to 'python'
where py >nul 2>&1 && set "PY=py" || set "PY=python"

echo.
echo [1/3] Checking/Installing required Python packages (first run may take a minute)...
%PY% -m pip install --upgrade pip >nul 2>&1
%PY% -m pip install flask pandas numpy python-pptx matplotlib >nul 2>&1

echo.
echo [2/3] Starting Update Helper on http://localhost:8777/ ...
cd /d "D:\Users\bclark\Tiktok\scripts"
%PY% "dashboard_update_server.py" "D:\Users\bclark\Tiktok\update_config.json"

echo.
echo [3/3] Server stopped. You can close this window.
pause
