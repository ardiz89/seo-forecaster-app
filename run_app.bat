@echo off
echo Starting SEO Forecaster...
cd /d "%~dp0"
call python -m streamlit run app.py
pause
