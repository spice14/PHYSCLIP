@echo off
REM Run PHYSCLIP v1 training script with output to current console
REM This ensures the script runs in the current window, not a new one

cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the training script with explicit python path from venv
"%cd%\venv\Scripts\python.exe" physclip/scripts/train_contrastive.py %*

REM Deactivate venv on exit
call venv\Scripts\deactivate.bat
