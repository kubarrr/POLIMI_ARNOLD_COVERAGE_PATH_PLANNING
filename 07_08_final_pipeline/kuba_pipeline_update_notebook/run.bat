@echo off
REM Windows convenience wrapper: forwards all arguments to the cross-platform
REM Python launcher. Example:
REM   run.bat --config paths.txt --method kmeans --k 3 --suplement nitrogen
python "%~dp0run.py" %*
