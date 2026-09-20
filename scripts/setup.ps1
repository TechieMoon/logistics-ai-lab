param([string]$Python = "python", [switch]$CpuOnly)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Assert-NativeSuccess { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    Assert-NativeSuccess
}
$Py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$Index = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
& $Py -m pip install torch==2.8.0 --index-url $Index
Assert-NativeSuccess
& $Py -m pip install -r requirements-lock-common.txt
Assert-NativeSuccess
& $Py -m pip install -e '.[dev]'
Assert-NativeSuccess
& $Py -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
Assert-NativeSuccess
