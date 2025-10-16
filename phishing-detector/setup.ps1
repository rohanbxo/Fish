# Quick Start Script for Phishing Detector
# This script helps you get started quickly

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phishing Detector - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "Setting up virtual environment..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "Virtual environment created!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Dependencies installed!" -ForegroundColor Green
Write-Host ""

# Create directories
Write-Host "Setting up directories..." -ForegroundColor Yellow
$dirs = @("logs", "results")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created $dir directory" -ForegroundColor Green
    }
}
Write-Host ""

# Prepare sample data
Write-Host "Preparing sample data..." -ForegroundColor Yellow
if (Test-Path "data\raw\sample_emails.csv") {
    python scripts\prepare_data.py
    Write-Host "Sample data prepared!" -ForegroundColor Green
} else {
    Write-Host "Sample data file not found. Skipping data preparation." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Start the API server:" -ForegroundColor White
Write-Host "   cd api; python main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "2. In a new terminal, start the web interface:" -ForegroundColor White
Write-Host "   streamlit run web\app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Run tests:" -ForegroundColor White
Write-Host "   pytest tests\ -v" -ForegroundColor Gray
Write-Host ""
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Web interface at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
