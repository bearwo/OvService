@echo off
echo ========================================
echo   OvService - Starting All Services
echo ========================================
echo.
echo Starting API server and Web UI...
echo API:    http://localhost:8000
echo Web UI: http://localhost:3000
echo.
python -u "%~dp0webui\server.py"
pause
