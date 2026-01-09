# DeepSeek-OCR Async - README

## What's New: Async Job Management

The async architecture solves the main problem with long PDF processing: **connection timeouts and no progress visibility**.

### The Problem (Before)

When processing a 200-page PDF:
- ❌ HTTP request blocks for 1+ hour
- ❌ Connection timeout loses all progress
- ❌ No way to know how far along you are
- ❌ Must stay connected the entire time
- ❌ If connection drops = start over

### The Solution (Now)

With async job management:
- ✅ Upload returns immediately (5 seconds)
- ✅ Get a job handle to track progress
- ✅ Check status anytime with real-time progress
- ✅ Disconnect and reconnect whenever you want
- ✅ Download results hours later if needed
- ✅ Jobs persist on server until cleaned up

## Quick Start

### 1. Deploy Async Server (Docker)

```powershell
# Replace server with async version
cp start_server_async.py start_server.py

# Create jobs directory
mkdir jobs

# Update docker-compose.yml to add jobs volume:
# volumes:
#   - ./jobs:/app/jobs

# Start server
docker-compose up --build -d
```

### 2. Run Async GUI

```powershell
.\venv\Scripts\Activate.ps1
python GUI_async.py
```

Access at: http://localhost:7862

### 3. Process a PDF

**Option A: Quick Process (GUI)**
1. Go to "Quick Process" tab
2. Upload your PDF
3. Click "Process and Wait"
4. Watch progress bar
5. Result appears automatically

**Option B: Manual Control (GUI)**
1. Go to "Manual Control" tab
2. Upload PDF → Get Job ID
3. Click "Check Status" periodically
4. When complete, click "Download Result"

**Option C: API (Python)**
```python
import requests
import time

# Create job
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/jobs/create",
        files={"file": ("document.pdf", f, "application/pdf")},
        data={"prompt": "<image>Convert to markdown"}
    )
    job_id = response.json()["job_id"]
    print(f"Job ID: {job_id}")

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
    print(f"{status['status']}: {status['progress']:.1f}% ({status['processed_pages']}/{status['total_pages']})")
    
    if status["status"] == "completed":
        break
    
    time.sleep(5)

# Download result
result = requests.get(f"http://localhost:8000/jobs/{job_id}/download")
with open("result.md", "wb") as f:
    f.write(result.content)
```

## How It Works

### Job Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. UPLOAD PDF                                               │
│    POST /jobs/create                                        │
│    → Returns job_id immediately (5 seconds)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. JOB QUEUED                                               │
│    Status: "queued"                                         │
│    Waiting for processing to start                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PROCESSING                                               │
│    Status: "processing"                                     │
│    Progress: 0% → 100%                                      │
│    GET /jobs/{job_id} to check progress                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. COMPLETED                                                │
│    Status: "completed"                                      │
│    GET /jobs/{job_id}/download to get result               │
└─────────────────────────────────────────────────────────────┘
```

### Job Storage

Each job creates a folder:
```
jobs/
  └── document.pdf_20260107_123456_a1b2c3d4/
      ├── input.pdf              # Your original PDF
      ├── metadata.json          # Job status
      ├── result.md              # Final markdown (when done)
      └── result_metadata.json   # Processing stats (when done)
```

Jobs auto-delete when:
- Age > 7 days (configurable)
- Total jobs > 1000 (configurable)
- Total storage > 10GB (configurable)

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/jobs/create` | POST | Upload PDF, get job ID |
| `/jobs/{job_id}` | GET | Check status and progress |
| `/jobs/{job_id}/download` | GET | Download completed result |
| `/jobs/{job_id}/metadata` | GET | Get processing metadata |
| `/jobs` | GET | List all recent jobs |
| `/jobs/{job_id}` | DELETE | Cancel a job |
| `/health` | GET | Server health + job stats |

## Configuration

### Environment Variables

```bash
# Job storage limits
MAX_JOB_STORAGE_MB=10240    # Max 10GB total
MAX_JOB_COUNT=1000          # Max 1000 jobs
MAX_JOB_AGE_HOURS=168       # Max 7 days old

# vLLM settings (same as before)
MAX_CONCURRENCY=1
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.95
# ... etc
```

### Docker Compose

```yaml
services:
  deepseek-ocr:
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./jobs:/app/jobs          # ADD THIS for job persistence
    environment:
      - MAX_JOB_STORAGE_MB=10240
      - MAX_JOB_COUNT=1000
      - MAX_JOB_AGE_HOURS=168
```

## Testing

### Automated Test

```powershell
# Run full test suite
python test_async_api.py path/to/test.pdf
```

This tests:
- ✅ Health check
- ✅ Job creation
- ✅ Status checking
- ✅ Progress polling
- ✅ Result download
- ✅ Metadata retrieval
- ✅ Job listing

### Manual Test

```powershell
# 1. Check health
curl http://localhost:8000/health

# 2. Create job
curl -X POST http://localhost:8000/jobs/create `
  -F "file=@test.pdf" `
  -F "prompt=<image>Convert to markdown"
# → Get job_id

# 3. Check status
curl http://localhost:8000/jobs/{job_id}

# 4. Download when done
curl http://localhost:8000/jobs/{job_id}/download -o result.md
```

## Examples

### Example 1: Process and Forget

```python
# Submit and walk away
job_id = create_job("huge_document.pdf")
print(f"Job submitted: {job_id}")
print("Come back later to download!")

# Hours later...
result = requests.get(f"{API}/jobs/{job_id}/download")
```

### Example 2: Progress Monitoring

```python
import time

job_id = create_job("document.pdf")

while True:
    status = get_job_status(job_id)
    
    pages = f"{status['processed_pages']}/{status['total_pages']}"
    print(f"[{status['status']}] {status['progress']:.1f}% - Pages: {pages}")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)
```

### Example 3: Batch Processing

```python
# Submit multiple jobs
job_ids = []
for pdf_file in pdf_files:
    job_id = create_job(pdf_file)
    job_ids.append(job_id)
    print(f"Submitted: {job_id}")

# Check all periodically
while job_ids:
    for job_id in job_ids[:]:
        status = get_job_status(job_id)
        if status['status'] == 'completed':
            download_result(job_id)
            job_ids.remove(job_id)
    
    time.sleep(10)
```

## Comparison

### Old Workflow
```python
# ❌ Blocks for entire duration
response = requests.post(
    "/ocr/pdf",
    files={"file": pdf_data},
    timeout=7200  # 2 hour timeout!
)
result = response.json()  # Hope it doesn't timeout!
```

### New Workflow
```python
# ✅ Returns immediately
response = requests.post(
    "/jobs/create",
    files={"file": pdf_data},
    timeout=30  # Only 30 seconds needed
)
job_id = response.json()["job_id"]

# ✅ Poll at your convenience
while not is_completed(job_id):
    time.sleep(10)

# ✅ Download anytime
result = download_result(job_id)
```

## Monitoring

### Check Server Status
```powershell
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_jobs": 1,      # Currently processing
  "queued_jobs": 3,      # Waiting to process
  "total_jobs": 157      # All jobs in system
}
```

### List All Jobs
```powershell
curl "http://localhost:8000/jobs?limit=50"
```

### Check Specific Job
```powershell
curl http://localhost:8000/jobs/{job_id}
```

## Troubleshooting

### Server Not Responding
```powershell
# Check if running
docker ps

# Check logs
docker-compose logs -f

# Restart
docker-compose restart
```

### Job Stuck
```powershell
# Check server logs for errors
docker-compose logs -f

# Job stuck in "processing" after restart → marked as failed
# Resubmit if needed
```

### Storage Full
```powershell
# Increase limits
$env:MAX_JOB_STORAGE_MB = "20480"  # 20GB

# Or clean manually
rm -r jobs/*
```

## Documentation

See detailed documentation:
- [`ASYNC_ARCHITECTURE.md`](ASYNC_ARCHITECTURE.md) - Full technical details
- [`ASYNC_QUICKSTART.md`](ASYNC_QUICKSTART.md) - Quick setup guide
- [`ASYNC_IMPLEMENTATION_SUMMARY.md`](ASYNC_IMPLEMENTATION_SUMMARY.md) - Implementation overview

## Migration

Switching from sync to async is easy:

**GUI Users:** Just use `GUI_async.py` instead of `GUI_enhanced.py`

**API Users:** See migration examples in [`ASYNC_ARCHITECTURE.md`](ASYNC_ARCHITECTURE.md)

## Benefits

| Feature | Before | After |
|---------|--------|-------|
| Upload time | Blocks until done | 5 seconds |
| Max document size | Limited by timeout | Unlimited |
| Connection loss | Lose everything | No problem |
| Progress visibility | None | Real-time |
| Reconnection | Not possible | Full support |
| Result storage | Immediate only | Configurable |

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed docs in `ASYNC_ARCHITECTURE.md`
3. Run the test suite: `python test_async_api.py`
4. Check server logs: `docker-compose logs -f`

---

**Ready to process long documents reliably? Start with `ASYNC_QUICKSTART.md`!** 🚀
