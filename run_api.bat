@echo off
REM Navigate to project root
cd /d %~dp0

REM Set Python path
set PYTHONPATH=%PYTHONPATH%;%cd%\code;%cd%

REM Run the API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info