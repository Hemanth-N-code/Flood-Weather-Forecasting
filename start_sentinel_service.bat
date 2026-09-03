@echo off
title Assam EdgeAI Sentinel - Autonomous Background Engine
cd /d "%~dp0"
echo ======================================================================
echo    Starting Assam Flood & Weather EdgeAI Prediction Daemon
echo    Interval: Every 60 minutes
echo    Cloud Sync: Firestore + Firebase Storage
echo ======================================================================
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe run_live_sentinel.py --daemon --interval 60
pause
