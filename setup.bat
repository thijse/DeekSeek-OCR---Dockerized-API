@echo off
echo ========================================
echo  DeepSeek-OCR Setup Script
echo ========================================
echo.

REM Check if DeepSeek-OCR directory exists
if not exist "DeepSeek-OCR" (
    echo DeepSeek-OCR directory not found. Cloning repository...
    git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
    if errorlevel 1 (
        echo Failed to clone DeepSeek-OCR repository. Make sure git is installed and accessible.
        pause
        exit /b 1
    )
    echo Repository cloned successfully.
) else (
    echo DeepSeek-OCR directory already exists.
)

REM Check if venv directory exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Make sure Python is installed and accessible.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip to latest version
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install huggingface_hub
echo Installing huggingface_hub...
pip install -U "huggingface_hub"
if errorlevel 1 (
    echo Failed to install huggingface_hub.
    pause
    exit /b 1
)
echo huggingface_hub installed successfully.

echo Downloading DeepSeek-OCR model from Hugging Face...
REM first create the models directory if it doesn't exist
if not exist "models\deepseek-ai\DeepSeek-OCR" (
    mkdir models\deepseek-ai\DeepSeek-OCR
)
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR

REM Install dependencies from requirements.txt
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
    echo Dependencies installed successfully.
) else (
    echo requirements.txt not found! Cannot install dependencies.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Build Docker:  build.bat
echo   2. Start Docker:  docker-compose up -d
echo   3. Start GUI:     python GUI.py
echo   4. Open browser:  http://localhost:7862
echo.

REM Check if we're already in an activated virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo Starting new command prompt with activated virtual environment...
    echo Type 'exit' to close this window.
    echo.
    cmd /k
) else (
    echo Virtual environment is already active in current session.
    pause
)

