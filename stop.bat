@echo off
echo Stopping UPI Intelligence Architecture...

REM Kill any streamlit process on port 8501
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| find ":8501"') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill lingering python processes running streamlit
wmic process where "commandline like '%%streamlit%%'" delete >nul 2>&1

echo Done. App has been stopped.
timeout /t 2 /nobreak >nul
