@echo off
setlocal
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m model_benchmark run %*
) else (
  py -3 -m model_benchmark run %*
)
