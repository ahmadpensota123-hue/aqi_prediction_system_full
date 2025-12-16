# 🐳 Docker Deployment Guide

This guide will help you manually build and run the AQI Prediction System using Docker.

## Prerequisites
- **Docker Desktop** installed and running.
- **Git** (optional, to pull code).
- Terminal (PowerShell on Windows, or Command Prompt).

---

## 🚀 Step-by-Step Instructions

### Step 1: Open Terminal
Navigate to the project directory:
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\aqi-prediction-system
```

### Step 2: Stop Local Services
If you have `uvicorn` or `streamlit` running locally, stop them (Ctrl+C) to free up ports `8000` and `8501`.

### Step 3: Run Docker Build
Execute the following command to build and start the containers.
> **Note:** The first build downloads ~500MB of data. On a slow connection, this may take 15-20 minutes. Please be patient!

```powershell
docker compose -f docker/docker-compose.yml up --build -d
```

**Flags explained:**
- `-f docker/docker-compose.yml`: Tells Docker where the config file is.
- `up`: Starts the containers.
- `--build`: Forces a rebuild of images (ensures latest code).
- `-d`: Detached mode (runs in background so it doesn't block your terminal).

### Step 4: Monitor Progress (Optional)
If you want to see the build logs (like I did), remove the `-d` flag:
```powershell
docker compose -f docker/docker-compose.yml up --build
```
*To stop monitoring but keep it running, you can't easily detach. Better to use `-d` and check logs later.*

### Step 5: Verify Deployment
Once the command finishes, check if containers are running:
```powershell
docker compose -f docker/docker-compose.yml ps
```
You should see `aqi-api` and `aqi-dashboard` with status "Up".

### Step 6: Access App
- **Dashboard:** [http://localhost:8501](http://localhost:8501)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ Troubleshooting

### "Build is stuck at apt-get update"
This is a network issue. Docker is trying to fetch Linux packages.
- **Solution:** Just wait. It can take 10+ minutes on restricted networks. It will eventually finish or timeout.

### "Ports are already allocated"
Error: `Bind for 0.0.0.0:8000 failed: port is already allocated`.
- **Solution:** You have the local Python app running. Stop it, or kill the process.

### "docker-compose not found"
- **Solution:** Use `docker compose` (with a space) instead of `docker-compose`.

### How to Stop Docker
To stop the containers:
```powershell
docker compose -f docker/docker-compose.yml down
```
