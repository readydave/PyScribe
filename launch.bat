@echo off
setlocal
cls

REM ===============================================================
REM PyScribe Launcher (Windows .bat)
REM Purpose:
REM   Start your app with the external Python 3.12 venv at:
REM     C:\Code\_envs\pyscribe\Scripts\python.exe
REM Notes:
REM   - Runs in the same window so you see errors and print output.
REM   - Shows the process exit code and pauses at the end.
REM   - Pass CLI args to this .bat and they’ll be forwarded to main.py.
REM ===============================================================

REM %~dp0 expands to the drive + path of THIS .bat file (with trailing slash).
REM Using it makes the launcher portable if you move the repo folder.
set "REPO_DIR=%~dp0"

REM Absolute path to the Python interpreter inside your external venv.
REM If you ever rename or relocate the venv, update this line only.
set "PYTHON=C:\Code\_envs\pyscribe\Scripts\python.exe"

REM The entry script you want to run. Keeping it as a variable is handy
REM if the app grows and you change the entry point later.
set "ENTRY=main.py"

REM ---------- Sanity checks (fail fast with helpful messages) ----------
if not exist "%PYTHON%" (
  echo [ERROR] Python not found at:
  echo   %PYTHON%
  echo Create it with:
  echo   py -3.12 -m venv C:\Code\_envs\pyscribe
  pause
  exit /b 1
)

if not exist "%REPO_DIR%%ENTRY%" (
  echo [ERROR] Entry script not found next to this launcher:
  echo   %REPO_DIR%%ENTRY%
  pause
  exit /b 1
)

REM Change into the repo directory so all relative paths in your code work.
pushd "%REPO_DIR%"

REM ---------- Run the app ----------
REM %* forwards any arguments you pass to launch.bat on to main.py.
REM Example: launch.bat --model small --verbose
"%PYTHON%" "%REPO_DIR%%ENTRY%" %*
set "RC=%ERRORLEVEL%"

REM Go back to the original folder we were in before pushd.
popd

REM Show the exit code so failures are obvious, then pause so the window
REM stays open and you can read output.
echo(
echo Exit code: %RC%
echo(
pause
endlocal

REM ===============================================================
REM Alternate: Run in a NEW console window that stays open
REM   Replace the run section above with this single line if desired:
REM
REM start "" cmd.exe /k ^"cd /d "%REPO_DIR%" ^&^& "%PYTHON%" "%REPO_DIR%%ENTRY%" %*^"
REM ===============================================================
