@echo off
echo Converting Markdown files to Word documents...
powershell -ExecutionPolicy Bypass -File ".\Simple-MDtoWord.ps1"
echo.
echo If the conversion was successful, you should now have .docx files for each .md file.
echo.
pause
