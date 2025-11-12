# Start the FastAPI inference server
# Set the port (default: 3000)
$env:PORT = if ($env:PORT) { $env:PORT } else { "3000" }

# Navigate to project root directory (parent of scripts folder)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "[start-server] DeepSeek-OCR API Server" -ForegroundColor Cyan
Write-Host "[start-server] Starting at $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')" -ForegroundColor Cyan
Write-Host "[start-server] Port: $env:PORT" -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[start-server] Activating virtual environment..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "[start-server] Warning: Virtual environment not found. Run '.\install-deps.ps1' first." -ForegroundColor Red
    exit 1
}

# Start the server using module syntax
Write-Host "[start-server] Starting FastAPI server..." -ForegroundColor Green
python -m main

