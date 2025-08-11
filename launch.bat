@echo off
CLS
ECHO Initializing PyScribe Environment...

REM This command starts a new command prompt window.
REM The '/k' flag tells the new window to execute the command that follows
REM and then remain open. This is useful for seeing any errors.

cmd.exe /k "cd /d %~dp0 && .venv\Scripts\activate && python main.py"
