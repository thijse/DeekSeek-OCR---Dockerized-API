@echo off
REM DeepSeek-OCR Docker Build Script for Windows
REM This script builds the Docker container

echo ========================================
echo  DeepSeek-OCR Docker Build Script
echo ========================================
echo.

REM Check if models directory exists
if not exist "models" (
    echo Warning: Models directory not found. Creating it...
    mkdir models
    echo Please download the DeepSeek-OCR model to models\deepseek-ai\DeepSeek-OCR\
    echo Run: huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR
    echo.
)

REM Check if model files exist
if not exist "models\deepseek-ai\DeepSeek-OCR\config.json" (
    echo ERROR: Model files not found in models\deepseek-ai\DeepSeek-OCR\
    echo Please download the model first:
    echo   huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR
    echo.
    pause
    exit /b 1
)

REM Check if DeepSeek-OCR source exists
if not exist "DeepSeek-OCR\DeepSeek-OCR-master" (
    echo ERROR: DeepSeek-OCR source not found in DeepSeek-OCR\DeepSeek-OCR-master\
    echo Please run setup.bat first to clone the repository.
    pause
    exit /b 1
)

REM Build the Docker image
echo Building Docker image...
echo This may take 10-20 minutes on first build...
echo.

docker-compose build

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED!
    echo.
    echo Possible solutions:
    echo   1. Ensure Docker Desktop is running with GPU support
    echo   2. Check that NVIDIA Container Toolkit is installed
    echo   3. Verify you have sufficient disk space (10GB+)
    echo   4. Try running: docker system prune -f
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Build Complete!
echo ========================================
echo.
echo To start the service:
echo   docker-compose up -d
echo.
echo To check the service:
echo   curl http://localhost:8000/health
echo.
echo To view logs:
echo   docker-compose logs -f deepseek-ocr
echo.
echo To start the web GUI:
echo   python GUI.py
echo   Then open http://localhost:7862
echo.
pause