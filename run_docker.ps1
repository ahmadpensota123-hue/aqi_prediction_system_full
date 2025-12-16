<#
.SYNOPSIS
    Helper script to deploy AQI System with Docker
.DESCRIPTION
    Automatically navigates to the project directory and runs docker compose.
#>

$ErrorActionPreference = "Stop"

# 1. Navigate to the script's directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Navigating to project folder: $ScriptDir" -ForegroundColor Cyan
Set-Location -Path $ScriptDir

# 2. Check for existing local processes
$RunningLocal = Get-Process -Name "uvicorn", "streamlit" -ErrorAction SilentlyContinue
if ($RunningLocal) {
    Write-Host "Warning: Local 'uvicorn' or 'streamlit' processes found." -ForegroundColor Yellow
    Write-Host "If ports 8000/8501 are busy, the docker container might fail to start." -ForegroundColor Yellow
}

# 3. Run Docker Compose
Write-Host "Starting Docker Build & Deploy..." -ForegroundColor Green
docker compose -f docker/docker-compose.yml up --build -d

# 4. Show status
Write-Host "Command executed. Checking container status..." -ForegroundColor Green
Start-Sleep -Seconds 5
docker compose -f docker/docker-compose.yml ps

Write-Host "`nDashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "API Docs:  http://localhost:8000/docs" -ForegroundColor Cyan
