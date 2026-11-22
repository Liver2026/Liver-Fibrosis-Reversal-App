@echo off
cd /d "%~dp0"
python FRM_MTEM_Desktop_App.py
if errorlevel 1 pause
