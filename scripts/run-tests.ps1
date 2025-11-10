# Run tests for DeepSeek-OCR API
# Usage:
#   .\run-tests.ps1              # Run all tests including integration
#   .\run-tests.ps1 -Unit        # Run unit tests only (skip integration)
#   .\run-tests.ps1 -Coverage    # Run with coverage report

param(
    [switch]$Unit,
    [switch]$Coverage,
    [switch]$Verbose
)

# Navigate to project root directory (parent of scripts folder)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "[run-tests] DeepSeek-OCR API - Test Runner" -ForegroundColor Cyan
Write-Host "[run-tests] Starting at $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')" -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[run-tests] Activating virtual environment..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "[run-tests] Error: Virtual environment not found. Run '.\install-deps.ps1' first." -ForegroundColor Red
    exit 1
}

# Check if pytest is installed
Write-Host "[run-tests] Checking dependencies..." -ForegroundColor Yellow
$pytestInstalled = python -c "import pytest; print('OK')" 2>$null
if ($pytestInstalled -ne "OK") {
    Write-Host "[run-tests] Installing test dependencies..." -ForegroundColor Yellow
    pip install pytest pytest-asyncio pytest-cov httpx
}

# Build pytest command
$pytestArgs = @()

# Add verbose flag if requested
if ($Verbose) {
    $pytestArgs += "-vv"
} else {
    $pytestArgs += "-v"
}

# Add marker for unit tests only
if ($Unit) {
    Write-Host "[run-tests] Running unit tests only (skipping integration tests)..." -ForegroundColor Yellow
    $pytestArgs += "-m"
    $pytestArgs += "not integration"
} else {
    Write-Host "[run-tests] Running all tests (including integration tests)..." -ForegroundColor Yellow
    Write-Host "[run-tests] Note: Integration tests require GPU and will load the OCR model." -ForegroundColor Yellow
}

# Add coverage options if requested
if ($Coverage) {
    Write-Host "[run-tests] Running with coverage report..." -ForegroundColor Yellow
    $pytestArgs += "--cov=."
    $pytestArgs += "--cov-report=term-missing"
    $pytestArgs += "--cov-report=html"
}

# Add test path
$pytestArgs += "tests/"

# Run pytest
Write-Host "[run-tests] Executing: pytest $($pytestArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""

python -m pytest @pytestArgs

$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[run-tests] All tests passed!" -ForegroundColor Green
    if ($Coverage) {
        Write-Host "[run-tests] Coverage report generated in 'htmlcov/index.html'" -ForegroundColor Green
    }
} else {
    Write-Host "[run-tests] Some tests failed. Exit code: $exitCode" -ForegroundColor Red
}

Write-Host "[run-tests] Completed at $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')" -ForegroundColor Cyan

exit $exitCode


