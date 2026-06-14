@echo off
title PlagCheck AI - Local Server Launcher
echo =========================================================
echo          PlagCheck AI - Local Server Launcher
echo =========================================================
echo.
echo [1/3] Installing / verifying Python libraries...
pip install -r requirements.txt --quiet
echo.
echo [2/3] Verifying MongoDB connection and seeding default data...
python database.py
echo.
echo [3/3] Starting FastAPI Web Application...
echo.
echo  Server URL: http://127.0.0.1:8000
echo  Open the above URL in your browser to access PlagCheck AI.
echo  Press Ctrl+C to stop the server.
echo.
echo =========================================================
echo  NOTE: Make sure MongoDB is running on localhost:27017
echo =========================================================
echo.
python main.py
pause
