@echo off
echo ========================================
echo   OvService - CLI Chat
echo ========================================
echo.
set OPENVINO_LIB_PATHS=D:\AISpace\Tools\openvino_genai\runtime\bin\intel64\Release;D:\AISpace\Tools\openvino_genai\runtime\3rdparty\tbb\bin
python -u "%~dp0cli.py"
pause
