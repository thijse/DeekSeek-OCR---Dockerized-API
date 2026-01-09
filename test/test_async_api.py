#!/usr/bin/env python3
"""
Test script for async DeepSeek-OCR API

Tests all endpoints and verifies job lifecycle.
"""

import requests
import time
import json
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/health")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
        print("✅ Health check passed")
        return True
    else:
        print("❌ Health check failed")
        return False

def test_create_job(pdf_path):
    """Test job creation"""
    print("\n" + "=" * 60)
    print("TEST 2: Create Job")
    print("=" * 60)
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
        data = {'prompt': '<image>Convert the content of the image to markdown.'}
        
        response = requests.post(f"{API_BASE}/jobs/create", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            job_id = result.get('job_id')
            print(f"✅ Job created: {job_id}")
            return job_id
        else:
            print("❌ Job creation failed")
            return None
    else:
        print(f"❌ Request failed: {response.text}")
        return None

def test_job_status(job_id):
    """Test job status endpoint"""
    print("\n" + "=" * 60)
    print("TEST 3: Check Job Status")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/jobs/{job_id}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
        print(f"✅ Status check passed")
        return data
    else:
        print(f"❌ Status check failed: {response.text}")
        return None

def test_poll_until_complete(job_id, max_wait=3600):
    """Poll job until completion"""
    print("\n" + "=" * 60)
    print("TEST 4: Poll Until Complete")
    print("=" * 60)
    
    start_time = time.time()
    last_progress = -1
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_BASE}/jobs/{job_id}")
        
        if response.status_code != 200:
            print(f"❌ Status check failed: {response.status_code}")
            return False
        
        data = response.json()
        status = data.get('status')
        progress = data.get('progress', 0)
        processed = data.get('processed_pages', 0)
        total = data.get('total_pages', 0)
        
        # Print progress updates
        if progress != last_progress:
            print(f"[{time.time() - start_time:.1f}s] Status: {status}, Progress: {progress:.1f}%, Pages: {processed}/{total}")
            last_progress = progress
        
        if status == 'completed':
            print(f"✅ Job completed in {time.time() - start_time:.1f}s")
            return True
        elif status == 'failed':
            error = data.get('error', 'Unknown error')
            print(f"❌ Job failed: {error}")
            return False
        elif status == 'cancelled':
            print(f"❌ Job was cancelled")
            return False
        
        time.sleep(2)
    
    print(f"❌ Timeout waiting for job completion")
    return False

def test_download_result(job_id, output_path="test_result.md"):
    """Test result download"""
    print("\n" + "=" * 60)
    print("TEST 5: Download Result")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/jobs/{job_id}/download")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = len(response.content)
        print(f"Downloaded {file_size} bytes to {output_path}")
        
        # Show first 500 characters
        content = response.content.decode('utf-8')
        print("\nFirst 500 characters of result:")
        print("-" * 60)
        print(content[:500])
        print("-" * 60)
        
        print(f"✅ Download successful")
        return True
    else:
        print(f"❌ Download failed: {response.text}")
        return False

def test_get_metadata(job_id):
    """Test metadata retrieval"""
    print("\n" + "=" * 60)
    print("TEST 6: Get Metadata")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/jobs/{job_id}/metadata")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
        print(f"✅ Metadata retrieval successful")
        return True
    else:
        print(f"❌ Metadata retrieval failed: {response.text}")
        return False

def test_list_jobs():
    """Test job listing"""
    print("\n" + "=" * 60)
    print("TEST 7: List Jobs")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/jobs?limit=10")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        jobs = data.get('jobs', [])
        print(f"Found {len(jobs)} jobs")
        
        for i, job in enumerate(jobs[:5]):  # Show first 5
            print(f"\nJob {i+1}:")
            print(f"  ID: {job.get('job_id')}")
            print(f"  Filename: {job.get('filename')}")
            print(f"  Status: {job.get('status')}")
            print(f"  Progress: {job.get('progress', 0):.1f}%")
        
        print(f"✅ Job listing successful")
        return True
    else:
        print(f"❌ Job listing failed: {response.text}")
        return False

def run_full_test(pdf_path):
    """Run all tests in sequence"""
    print("\n" + "🧪" * 30)
    print("ASYNC OCR API TEST SUITE")
    print("🧪" * 30 + "\n")
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health()))
    
    if not results[0][1]:
        print("\n❌ Health check failed, aborting tests")
        return
    
    # Test 2: Create job
    job_id = test_create_job(pdf_path)
    results.append(("Create Job", job_id is not None))
    
    if not job_id:
        print("\n❌ Job creation failed, aborting tests")
        print_summary(results)
        return
    
    # Test 3: Check status
    status_data = test_job_status(job_id)
    results.append(("Check Status", status_data is not None))
    
    # Test 4: Poll until complete
    completed = test_poll_until_complete(job_id)
    results.append(("Poll Until Complete", completed))
    
    if completed:
        # Test 5: Download result
        results.append(("Download Result", test_download_result(job_id)))
        
        # Test 6: Get metadata
        results.append(("Get Metadata", test_get_metadata(job_id)))
    
    # Test 7: List jobs
    results.append(("List Jobs", test_list_jobs()))
    
    # Summary
    print_summary(results)

def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Look for any PDF in data/uploads
        upload_dir = Path("data/uploads")
        if upload_dir.exists():
            pdfs = list(upload_dir.glob("*.pdf"))
            if pdfs:
                pdf_path = str(pdfs[0])
                print(f"Using PDF: {pdf_path}")
            else:
                print("❌ No PDF files found in data/uploads/")
                print("\nUsage: python test_async_api.py [path/to/test.pdf]")
                sys.exit(1)
        else:
            print("❌ data/uploads/ directory not found")
            print("\nUsage: python test_async_api.py [path/to/test.pdf]")
            sys.exit(1)
    
    # Run tests
    run_full_test(pdf_path)
