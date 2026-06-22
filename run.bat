@echo off
REM ===================================================================
REM  NeuroPredict - one-command launcher (Windows)
REM
REM  Just double-click this file, or run:  run.bat
REM
REM  It sets up everything by itself and then opens the website:
REM    1. creates a private Python environment (.venv)
REM    2. installs the libraries the app needs
REM    3. trains the small demo model the first time (a few minutes)
REM    4. starts the website at http://localhost:8000 and opens your browser
REM
REM  Works offline once the libraries are installed - no account, no
REM  token, and no Devin required.
REM ===================================================================

setlocal
cd /d "%~dp0"

if "%PORT%"=="" set "PORT=8000"
set "URL=http://localhost:%PORT%"

REM 1. Find Python (prefer "python", fall back to the "py" launcher).
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py"
if not defined PY (
  echo ERROR: Python 3 is not installed.
  echo Install it from https://www.python.org/downloads/
  echo During install, tick "Add Python to PATH", then run this again.
  pause
  exit /b 1
)

REM 2. Create the private environment the first time.
if not exist ".venv" (
  echo ==^> Creating Python environment ^(.venv^) ...
  %PY% -m venv .venv
)

set "VENV_PY=.venv\Scripts\python.exe"

echo ==^> Installing libraries ^(first run can take a few minutes^) ...
"%VENV_PY%" -m pip install --quiet --upgrade pip
"%VENV_PY%" -m pip install --quiet -r requirements.txt

REM 3. Train the demo model the first time (only if it's missing).
if not exist "models\wmd_multimodal.pt" (
  echo ==^> Training the demo model ^(first run only, a few minutes^) ...
  "%VENV_PY%" scripts\train_demo.py
) else (
  echo ==^> Demo model already present, skipping training.
)

REM 4. Open the browser, then start the website.
echo.
echo ==^> Starting NeuroPredict at %URL%
echo     ^(leave this window open while you use the site; press Ctrl+C to stop^)
echo.
start "" "%URL%"
"%VENV_PY%" -m uvicorn webapp.main:app --host 0.0.0.0 --port %PORT%

endlocal
