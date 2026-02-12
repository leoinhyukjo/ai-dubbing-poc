@echo off
REM AI Dubbing PoC Setup Script for Windows

echo ========================================
echo AI Dubbing PoC - Setup Script (Windows)
echo ========================================
echo.

REM Check Python
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please install Python 3.10 or higher from https://python.org
    pause
    exit /b 1
)
python --version
echo.

REM Check FFmpeg
echo Checking FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ERROR: FFmpeg not found
    echo.
    echo Please install FFmpeg:
    echo 1. Download from https://ffmpeg.org/download.html
    echo 2. Extract to a folder
    echo 3. Add the bin folder to your PATH
    pause
    exit /b 1
)
ffmpeg -version 2>&1 | findstr /C:"ffmpeg version"
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo WARNING: Virtual environment already exists
    set /p recreate="Remove and recreate? (y/N): "
    if /i "%recreate%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo Virtual environment recreated
    )
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed
echo.

REM Create directories
echo Creating directories...
if not exist temp mkdir temp
if not exist output mkdir output
if not exist samples mkdir samples
echo Directories created
echo.

REM Setup .env
echo Setting up environment file...
if not exist .env (
    copy .env.example .env
    echo .env file created from template
    echo.
    echo IMPORTANT: Edit .env file and add your API keys:
    echo    - OPENAI_API_KEY
    echo    - ELEVENLABS_API_KEY
) else (
    echo .env file already exists
)
echo.

echo ========================================
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file and add your API keys
echo 2. Prepare audio samples in 'samples\' directory
echo 3. Run: python examples\create_voice_clone.py
echo 4. Run: python pipeline.py your_video.mp4 VOICE_ID
echo.
echo For more information, see README.md
echo ========================================
pause
