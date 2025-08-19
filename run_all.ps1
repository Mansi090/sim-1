# Runs the full stack: server -> open simulator -> navigator
# Usage (PowerShell):
#   .\run_all.ps1
# If you get script execution errors, run once:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = 'Stop'

function Require-VenvPython {
  $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
  if (-not (Test-Path $venvPy)) {
    Write-Host "[run_all] Creating virtual environment..."
    python -m venv (Join-Path $PSScriptRoot ".venv")
  }
  return $venvPy
}

function Ensure-Requirements($venvPy) {
  Write-Host "[run_all] Upgrading pip/wheel/setuptools..."
  & $venvPy -m pip install --upgrade pip wheel setuptools | Out-Host
  Write-Host "[run_all] Installing requirements.txt..."
  & $venvPy -m pip install -r (Join-Path $PSScriptRoot "requirements.txt") | Out-Host
}

function Start-Server($venvPy) {
  Write-Host "[run_all] Starting server.py (new window)..."
  Start-Process -FilePath $venvPy -ArgumentList "server.py" -WorkingDirectory $PSScriptRoot -WindowStyle Normal | Out-Null
}

function Open-SimulatorPage {
  Write-Host "[run_all] Starting static HTTP server on http://127.0.0.1:8000 ..."
  Start-Process -FilePath (Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe") -ArgumentList "-m http.server 8000" -WorkingDirectory $PSScriptRoot -WindowStyle Hidden | Out-Null
  Start-Sleep -Seconds 1
  Write-Host "[run_all] Opening simulator page at http://127.0.0.1:8000/index.html ..."
  Start-Process "http://127.0.0.1:8000/index.html" | Out-Null
}

function Wait-For-WebSocket {
  # Wait for ws server to come up on localhost:8080 using Test-NetConnection
  Write-Host "[run_all] Waiting for WebSocket ws://localhost:8080 ..."
  $maxWait = 20
  for ($i = 0; $i -lt $maxWait; $i++) {
    try {
      $res = Test-NetConnection -ComputerName localhost -Port 8080 -WarningAction SilentlyContinue
      if ($res.TcpTestSucceeded) { Write-Host "[run_all] WebSocket is up."; return }
    } catch { }
    Start-Sleep -Seconds 1
  }
  Write-Warning "[run_all] WebSocket may not be up yet; proceeding anyway."
}

function Run-Navigator($venvPy) {
  Write-Host "[run_all] Running Level 1 (four corners) navigator..."
  & $venvPy (Join-Path $PSScriptRoot "auto_navigator_level1.py")
}

$venvPy = Require-VenvPython
Ensure-Requirements $venvPy
Start-Server $venvPy
Open-SimulatorPage
Wait-For-WebSocket
Run-Navigator $venvPy
