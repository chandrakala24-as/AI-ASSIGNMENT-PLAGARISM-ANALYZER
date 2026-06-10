@echo off
title AI Assignment Plagiarism Analyser Launcher
echo =========================================================
echo       AI Assignment Plagiarism Analyser Launcher
echo =========================================================
echo.
echo [1/3] Checking python libraries...
pip install -r requirements.txt
echo.
echo [2/3] Initializing SQLite database schemas...
python database.py
echo.
echo [3/3] Launching FastAPI Web Application...
echo Server starting on http://127.0.0.1:8000
echo Open http://127.0.0.1:8000 in your browser.
echo Press Ctrl+C in this terminal window to stop the server.
echo.
python main.py
pause
