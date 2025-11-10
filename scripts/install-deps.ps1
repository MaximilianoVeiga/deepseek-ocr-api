# Install dependencies for DeepSeek-OCR API
# Navigate to project root directory (parent of scripts folder)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "[install-deps] DeepSeek-OCR API - Dependency Installation" -ForegroundColor Cyan
Write-Host "[install-deps] Starting at $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')" -ForegroundColor Cyan

# Create and activate virtual environment if it doesn't exist
if (-not (Test-Path ".\.venv")) {
    Write-Host "[install-deps] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "[install-deps] Activating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "[install-deps] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA support first
Write-Host "[install-deps] Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies from requirements.txt
Write-Host "[install-deps] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "[install-deps] Dependencies installed successfully!" -ForegroundColor Green
Write-Host "[install-deps] Run '.\start-server.ps1' to start the API server." -ForegroundColor Green

