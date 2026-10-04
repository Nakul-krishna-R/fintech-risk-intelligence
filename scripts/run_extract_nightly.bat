@echo off
REM Runs the risk extraction over any not-yet-processed chunks in chunks.csv.
REM Triggered nightly by the "FintechRiskExtraction" Windows Scheduled Task.
REM Resumable: safe to let this run partially and pick up again another night.

cd /d "C:\Users\NAKUL KRISHNA R\fintech-risk-intelligence"
if not exist logs mkdir logs

echo ---- run started %date% %time% ---- >> logs\extract_nightly.log
"C:\Users\NAKUL KRISHNA R\AppData\Local\Programs\Python\Python312\python.exe" -u src\extract.py >> logs\extract_nightly.log 2>&1
echo ---- run finished %date% %time% (exit code %errorlevel%) ---- >> logs\extract_nightly.log
