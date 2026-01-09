# Async Architecture Improvement Plan

## Current Issues
1. **Synchronous blocking**: GUI waits for entire PDF processing (can take hours for large docs)
2. **No progress tracking**: Client can't query actual server-side progress
3. **Connection timeouts**: Long requests can timeout with no recovery
4. **No persistence**: If connection drops, work is lost
5. **No result retrieval**: Can't download results after processing completes

## Proposed Architecture

### Server-Side Changes (start_server.py)

#### 1. Job Management System
- **Job Storage**: Store jobs with status, progress, and results on disk
  - Directory: `/app/jobs/` with subdirectories:
    - `pending/` - uploaded files waiting to process
    - `processing/` - currently being processed
    - `completed/` - finished jobs with results
    - `failed/` - jobs that encountered errors
  
- **Job ID Format**: `{sanitized_filename}_{timestamp}_{uuid4}`
  - Example: `my_document_pdf_20260107_143022_a7b3c4d5`

- **Job Metadata** (JSON file per job):
  ```json
  {
    "job_id": "...",
    "filename": "original.pdf",
    "status": "pending|processing|completed|failed",
    "created_at": "2026-01-07T14:30:22Z",
    "started_at": "2026-01-07T14:30:25Z",
    "completed_at": "2026-01-07T14:45:10Z",
    "total_pages": 150,
    "processed_pages": 150,
    "progress_percent": 100,
    "prompt": "...",
    "result_file": "result.md",
    "error": null
  }
  ```

#### 2. New API Endpoints

**POST /api/upload**
- Upload PDF and get job ID immediately
- Returns: `{"job_id": "...", "status": "pending", "total_pages": 150}`

**GET /api/jobs/{job_id}/status**
- Get current status and progress
- Returns: `{"status": "processing", "progress": 65, "pages_done": 98, "total_pages": 150}`

**GET /api/jobs/{job_id}/result**
- Download completed result (markdown file)
- Returns: File download or 404 if not ready

**GET /api/jobs**
- List all jobs (with pagination)
- Returns: `{"jobs": [...]}`

**DELETE /api/jobs/{job_id}**
- Delete a job and its files
- Returns: `{"deleted": true}`

#### 3. Background Processing
- Use FastAPI BackgroundTasks to process PDFs asynchronously
- Update job status and progress in real-time
- Store results persistently

#### 4. Storage Management
- Track total storage used
- Implement LRU (Least Recently Used) cleanup
- Configurable limits:
  - `MAX_JOBS`: 100 jobs
  - `MAX_STORAGE_GB`: 10 GB
  - Auto-purge oldest completed jobs when limits reached

### Client-Side Changes (GUI_enhanced.py)

#### 1. Upload & Poll Pattern
- Upload file → get job_id
- Poll `/api/jobs/{job_id}/status` every 2-5 seconds
- Update progress bar in real-time
- Download result when complete

#### 2. Reconnection Support
- Store job_ids locally
- Allow user to reconnect to running jobs
- Show list of recent jobs with status

#### 3. Better UX
- Real progress from server (not simulated)
- Ability to start multiple jobs
- Download results anytime
- Clear error messages

## Implementation Steps

### Phase 1: Server Infrastructure
1. Create job storage system with disk persistence
2. Implement job metadata management
3. Add storage tracking and cleanup

### Phase 2: Server API
1. Implement `/api/upload` endpoint
2. Implement `/api/jobs/{job_id}/status` endpoint
3. Implement `/api/jobs/{job_id}/result` endpoint
4. Add background processing with progress updates
5. Implement `/api/jobs` list endpoint
6. Add `/api/jobs/{job_id}` delete endpoint

### Phase 3: Client Updates
1. Update upload logic to use new async API
2. Implement polling mechanism
3. Add reconnection UI
4. Update progress display

### Phase 4: Storage Management
1. Implement LRU cleanup
2. Add storage limits
3. Add admin endpoints for cleanup

## File Structure

```
/app/
  jobs/
    {job_id}/
      input.pdf          # Original uploaded file
      metadata.json      # Job status and metadata
      result.md          # Generated markdown (when complete)
      progress.json      # Real-time progress updates
  start_server.py        # Enhanced with job management
```

## Benefits

1. **Resilience**: Connection drops don't lose work
2. **Progress**: Real server-side progress tracking
3. **Scalability**: Multiple concurrent jobs
4. **User Experience**: No long blocking waits
5. **Flexibility**: Can close browser and return later
6. **Debugging**: Job history and logs available
