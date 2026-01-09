# Quick Start: Async DeepSeek-OCR

## Option 1: Docker Deployment (Recommended)

### Step 1: Prepare Files

```powershell
# Replace the default server with async version
cp start_server_async.py start_server.py
```

### Step 2: Update docker-compose.yml

Add job storage volume:

```yaml
services:
  deepseek-ocr:
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./jobs:/app/jobs        # ADD THIS LINE
    environment:
      # ... existing settings ...
      - MAX_JOB_STORAGE_MB=10240  # Optional: 10GB limit
      - MAX_JOB_COUNT=1000         # Optional: max 1000 jobs
      - MAX_JOB_AGE_HOURS=168      # Optional: 7 days retention
```

### Step 3: Create Jobs Directory

```powershell
mkdir jobs
```

### Step 4: Rebuild and Start

```powershell
# Stop existing container
docker-compose down

# Rebuild with async server
docker-compose up --build -d

# Check logs
docker-compose logs -f
```

### Step 5: Test with Async GUI

```powershell
# Activate Python environment
.\venv\Scripts\Activate.ps1

# Run async GUI
python GUI_async.py
```

Access at: http://localhost:7862

---

## Option 2: Local Development (No Docker)

### Step 1: Ensure vLLM is Installed

The async server requires the same vLLM setup as the original. If you haven't set this up, you need the DeepSeek-OCR model and vLLM environment.

### Step 2: Create Jobs Directory

```powershell
mkdir jobs
```

### Step 3: Run Async Server

```powershell
.\venv\Scripts\Activate.ps1

# Set environment variables (adjust paths as needed)
$env:MODEL_PATH = ".\models\deepseek-ai\DeepSeek-OCR"
$env:MAX_JOB_STORAGE_MB = "10240"
$env:MAX_JOB_COUNT = "1000"

# Run server
python start_server_async.py
```

### Step 4: Run Async GUI (in another terminal)

```powershell
.\venv\Scripts\Activate.ps1
python GUI_async.py
```

---

## Quick Test

### Test with cURL

1. **Create a job:**
```powershell
curl -X POST "http://localhost:8000/jobs/create" `
  -F "file=@data/uploads/your_document.pdf" `
  -F "prompt=<image>Convert to markdown"
```

Response:
```json
{
  "success": true,
  "job_id": "your_document.pdf_20260107_123456_a1b2c3d4",
  "message": "Job created successfully. Total pages: 50"
}
```

2. **Check status:**
```powershell
curl "http://localhost:8000/jobs/your_document.pdf_20260107_123456_a1b2c3d4"
```

Response:
```json
{
  "job_id": "your_document.pdf_20260107_123456_a1b2c3d4",
  "status": "processing",
  "progress": 45.5,
  "processed_pages": 23,
  "total_pages": 50
}
```

3. **Download when complete:**
```powershell
curl "http://localhost:8000/jobs/your_document.pdf_20260107_123456_a1b2c3d4/download" `
  -o result.md
```

---

## Comparison: Old vs New

### Old Synchronous Flow
```
1. Upload PDF via GUI
2. Wait 30+ minutes (blocking)
3. If connection drops → LOSE EVERYTHING
4. Get result (if lucky)
```

### New Async Flow
```
1. Upload PDF → Get job ID (5 seconds)
2. Close browser, go for coffee ☕
3. Come back anytime
4. Check status → 100% complete!
5. Download result
```

---

## Using Both Versions

You can run both servers simultaneously on different ports:

### Terminal 1: Sync Server (port 8000)
```powershell
docker-compose up
```

### Terminal 2: Async Server (port 8001)
```powershell
# Modify start_server_async.py to use port 8001
# Then run locally or in another Docker container
```

### GUI Selection
```powershell
# Sync version (old)
python GUI_enhanced.py

# Async version (new)  
python GUI_async.py
```

---

## Monitoring

### Check Server Health
```powershell
curl http://localhost:8000/health
```

Response shows:
- Model status
- Active jobs count
- Queued jobs count
- Total jobs

### List All Jobs
```powershell
curl "http://localhost:8000/jobs?limit=50"
```

### Job Storage Location

- **Docker**: Inside container at `/app/jobs/`
  - Mapped to `./jobs/` on host (if volume mounted)

- **Local**: `./jobs/` in project directory

Each job folder contains:
```
jobs/
  └── document.pdf_20260107_123456_a1b2c3d4/
      ├── input.pdf
      ├── metadata.json
      ├── result.md
      └── result_metadata.json
```

---

## Troubleshooting

### Server won't start
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <process_id> /F
```

### Jobs directory not created
```powershell
# Docker: Add volume mount in docker-compose.yml
# Local: Create manually
mkdir jobs
```

### Old jobs filling up disk
```powershell
# Reduce retention limits
$env:MAX_JOB_AGE_HOURS = "24"  # Only keep 1 day
$env:MAX_JOB_STORAGE_MB = "5120"  # Max 5GB

# Or manually clean
rm -r jobs/*
```

### Job stuck in "processing"
```powershell
# Check server logs for errors
docker-compose logs -f

# Or restart server (job will be marked as failed)
docker-compose restart
```

---

## Next Steps

1. ✅ Deploy async server
2. ✅ Test with small PDF
3. ✅ Test with large PDF (100+ pages)
4. ✅ Verify reconnection works (close GUI and reopen)
5. ✅ Check job history
6. ✅ Configure storage limits for your use case

---

## Production Recommendations

For production use:

```yaml
environment:
  # Keep 30 days of jobs
  - MAX_JOB_AGE_HOURS=720
  
  # Allow up to 50GB storage
  - MAX_JOB_STORAGE_MB=51200
  
  # Keep up to 5000 jobs
  - MAX_JOB_COUNT=5000
  
  # GPU settings for your hardware
  - MAX_CONCURRENCY=1
  - MAX_MODEL_LEN=2048
  - GPU_MEMORY_UTILIZATION=0.95
```

Consider:
- Regular backups of `./jobs/` directory
- Monitoring disk space
- Setting up alerts for failed jobs
- Adding authentication if exposing publicly

---

That's it! You now have a robust async OCR system that can handle long documents without connection issues. 🚀
