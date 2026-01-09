# DeepSeek-OCR Async Architecture

## Overview

The async architecture improves the handling of long PDF documents by implementing a job queue system that provides:

1. **Immediate Job Submission** - Upload your PDF and get a job handle immediately
2. **Progress Tracking** - Poll the API to check processing progress
3. **Reconnection Support** - Check status and download results even after disconnection
4. **Persistent Storage** - Jobs and results are stored on disk until storage limits
5. **Automatic Cleanup** - Old jobs are purged based on age, count, and storage size

## Architecture Components

### Server (`start_server_async.py`)

The async server extends the original `start_server.py` with job management capabilities:

#### Key Features

- **Job Queue Manager** - Manages job lifecycle and persistent storage
- **Background Processing** - Jobs process asynchronously without blocking
- **Persistent Storage** - Jobs saved to `/app/jobs/` directory (configurable)
- **Automatic Cleanup** - Removes old jobs based on configurable limits

#### Storage Limits

Configure via environment variables:
- `MAX_JOB_STORAGE_MB` - Max total storage in MB (default: 10240 = 10GB)
- `MAX_JOB_COUNT` - Max number of jobs to keep (default: 1000)
- `MAX_JOB_AGE_HOURS` - Max job age in hours (default: 168 = 7 days)

### Client (`GUI_async.py`)

The async GUI provides three usage modes:

1. **Quick Process** - Submit and auto-wait for completion
2. **Manual Control** - Full control over submit, check, download
3. **Job History** - View all recent jobs

## API Endpoints

### 1. Create Job
```
POST /jobs/create
Form Data:
  - file: PDF file (multipart/form-data)
  - prompt: Optional custom prompt (text)

Response:
{
  "success": true,
  "job_id": "document.pdf_20260107_123456_a1b2c3d4",
  "message": "Job created successfully. Total pages: 50"
}
```

### 2. Check Job Status
```
GET /jobs/{job_id}

Response:
{
  "job_id": "document.pdf_20260107_123456_a1b2c3d4",
  "filename": "document.pdf",
  "status": "processing",  // queued | processing | completed | failed | cancelled
  "progress": 45.5,        // 0.0 to 100.0
  "total_pages": 50,
  "processed_pages": 23,
  "created_at": 1704625200.123,
  "started_at": 1704625205.456,
  "completed_at": null,
  "error": null
}
```

### 3. Download Result
```
GET /jobs/{job_id}/download

Response: 
  - Content-Type: text/markdown
  - File download of result.md
```

### 4. Get Metadata
```
GET /jobs/{job_id}/metadata

Response:
{
  "filename": "document.pdf",
  "total_pages": 50,
  "processed_at": "2026-01-07T12:35:45",
  "processing_time_seconds": 750.2,
  "prompt_used": "<image>Convert to markdown"
}
```

### 5. List Jobs
```
GET /jobs?limit=100

Response:
{
  "jobs": [
    {
      "job_id": "...",
      "filename": "...",
      "status": "...",
      ...
    }
  ]
}
```

### 6. Cancel Job
```
DELETE /jobs/{job_id}

Response:
{
  "success": true,
  "message": "Job cancelled"
}
```

## Job Handle Format

Job IDs follow the pattern:
```
{clean_filename}_{timestamp}_{unique_id}
```

Example:
```
document.pdf_20260107_123456_a1b2c3d4
```

Where:
- `clean_filename` - Original filename (alphanumeric only)
- `timestamp` - Creation time (YYYYMMDD_HHMMSS)
- `unique_id` - 8-character UUID

## Storage Structure

Each job creates a directory:
```
/app/jobs/{job_id}/
  ├── input.pdf              # Original uploaded PDF
  ├── metadata.json          # Job status and metadata
  ├── result.md              # Final markdown result (when completed)
  └── result_metadata.json   # Processing metadata (when completed)
```

## Deployment

### Docker (Recommended)

The async server can run in the same Docker container. Update your docker-compose.yml:

```yaml
services:
  deepseek-ocr-async:
    build: .
    container_name: deepseek-ocr-async
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./jobs:/app/jobs          # Persistent job storage
    environment:
      - MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR
      - MAX_JOB_STORAGE_MB=10240  # 10GB
      - MAX_JOB_COUNT=1000
      - MAX_JOB_AGE_HOURS=168     # 7 days
      # ... other vLLM settings ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Build and run:
```bash
# Copy async server into Docker context
cp start_server_async.py start_server.py

# Build and start
docker-compose up --build -d
```

### Local Development

Run the async server locally:
```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Run async server (requires vLLM and DeepSeek-OCR setup)
python start_server_async.py
```

Run the async GUI:
```bash
# In another terminal
.\venv\Scripts\Activate.ps1
python GUI_async.py
```

## Usage Examples

### Python Client Example

```python
import requests
import time

API_BASE = "http://localhost:8000"

# 1. Upload and create job
with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    data = {"prompt": "<image>Convert to markdown"}
    
    response = requests.post(f"{API_BASE}/jobs/create", files=files, data=data)
    job_id = response.json()["job_id"]
    print(f"Created job: {job_id}")

# 2. Poll for completion
while True:
    response = requests.get(f"{API_BASE}/jobs/{job_id}")
    status_data = response.json()
    
    print(f"Status: {status_data['status']} - {status_data['progress']:.1f}%")
    
    if status_data["status"] in ["completed", "failed", "cancelled"]:
        break
    
    time.sleep(2)

# 3. Download result if completed
if status_data["status"] == "completed":
    response = requests.get(f"{API_BASE}/jobs/{job_id}/download")
    with open("result.md", "wb") as f:
        f.write(response.content)
    print("Result downloaded!")
```

### cURL Examples

Create job:
```bash
curl -X POST "http://localhost:8000/jobs/create" \
  -F "file=@document.pdf" \
  -F "prompt=<image>Convert to markdown"
```

Check status:
```bash
curl "http://localhost:8000/jobs/{job_id}"
```

Download result:
```bash
curl "http://localhost:8000/jobs/{job_id}/download" -o result.md
```

List jobs:
```bash
curl "http://localhost:8000/jobs?limit=10"
```

## Migration from Synchronous API

### For GUI Users

Simply switch from `GUI_enhanced.py` to `GUI_async.py`. The interface is similar but provides:
- Better responsiveness (no blocking)
- Ability to check old jobs
- Progress tracking during processing

### For API Users

**Old Synchronous API:**
```python
# Single blocking request
response = requests.post(
    "http://localhost:8000/ocr/pdf",
    files={"file": pdf_data},
    timeout=7200  # 2 hour timeout!
)
result = response.json()["results"]
```

**New Async API:**
```python
# 1. Quick submit
response = requests.post(
    "http://localhost:8000/jobs/create",
    files={"file": pdf_data},
    timeout=30  # Only 30s needed!
)
job_id = response.json()["job_id"]

# 2. Poll periodically
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(5)

# 3. Download when ready
result = requests.get(f"http://localhost:8000/jobs/{job_id}/download").text
```

## Benefits

### Before (Synchronous)
- ❌ Long HTTP connections (hours for large PDFs)
- ❌ Connection timeouts lose all progress
- ❌ No progress visibility
- ❌ Must wait until completion
- ❌ Lost results if client disconnects

### After (Async)
- ✅ Immediate job submission (seconds)
- ✅ Resume from any point
- ✅ Real-time progress tracking
- ✅ Can disconnect and reconnect
- ✅ Results persist on server
- ✅ Multiple concurrent uploads possible (jobs queue)

## Troubleshooting

### Job Not Starting

Check server logs and API health:
```bash
curl http://localhost:8000/health
```

Look for:
- `active_jobs` - Jobs currently processing
- `queued_jobs` - Jobs waiting to start
- Jobs process sequentially by default (only 1 active at a time)

### Storage Full

Adjust limits in environment variables:
```bash
MAX_JOB_STORAGE_MB=20480  # Increase to 20GB
MAX_JOB_COUNT=2000        # Increase job count
```

Or manually clean old jobs:
```bash
# Jobs stored in /app/jobs/ (Docker) or ./jobs/ (local)
rm -rf /app/jobs/old_job_id
```

### Result Download Fails

Check job status first:
```bash
curl http://localhost:8000/jobs/{job_id}
```

Only completed jobs can be downloaded. Status must be `"completed"`.

## Performance Considerations

### Job Processing Rate

With default settings (MAX_CONCURRENCY=1):
- 1 job processes at a time
- Other jobs wait in queue
- ~15-30 seconds per page typical

For 100-page document:
- Processing time: ~25-50 minutes
- But submission takes <5 seconds!
- Client can disconnect and reconnect anytime

### Storage Management

Job cleanup happens automatically when:
1. Total storage exceeds `MAX_JOB_STORAGE_MB`
2. Job count exceeds `MAX_JOB_COUNT`
3. Job age exceeds `MAX_JOB_AGE_HOURS`

Oldest jobs purged first (FIFO).

## Future Enhancements

Possible improvements:
- [ ] Resume interrupted jobs (restart from last completed page)
- [ ] Webhook notifications when jobs complete
- [ ] Priority queue for urgent jobs
- [ ] User authentication and job ownership
- [ ] Parallel processing for multiple GPUs
- [ ] Incremental result streaming (download pages as they complete)
- [ ] Job scheduling (process at specific times)

## Conclusion

The async architecture makes DeepSeek-OCR robust for production use with long documents. Upload immediately, track progress, and download results - even hours later!
