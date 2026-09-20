param([int]$Port = 8768)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
& '.\.venv\Scripts\python.exe' -m uvicorn logistics_ai.api:app --host 127.0.0.1 --port $Port
