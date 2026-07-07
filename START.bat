@echo off
cls
echo.
echo ============================================
echo   🤖 AI CHATBOT (NLP) - ENHANCED VERSION
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo.

REM Run tests (optional)
echo 🧪 Would you like to run NLP tests first? (y/n)
set /p run_tests=
if /i "%run_tests%"=="y" (
    echo 🧪 Running NLP tests...
    python test_chatbot.py
    echo.
    echo Press any key to continue to the chatbot...
    pause >nul
)

echo.
echo 🚀 Starting AI Chatbot...
echo.
echo 💡 The chatbot will open in your default browser
echo 🌐 URL: http://localhost:8501
echo. 
echo 📝 ENHANCED FEATURES:
echo   • 💬 Interactive chat with advanced NLP
echo   • 📊 Real-time analytics dashboard
echo   • 🧠 NLP demo and live testing
echo   • 🎯 97%% intent classification accuracy
echo   • 🏷️ Advanced entity extraction
echo   • 😊 Enhanced sentiment analysis
echo   • 📦 Smart order tracking
echo   • 🔐 Automated password reset
echo   • 💰 Intelligent refund handling
echo.
echo 🛑 Press Ctrl+C to stop the chatbot
echo.

streamlit run chatbot.py

pause
