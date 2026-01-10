#!/usr/bin/env python3
"""
DeepSeek-OCR Docker Entrypoint
Starts the API server and optionally the GUI
"""

import os
import sys
import subprocess
import time

def main():
    print("=" * 40)
    print(" DeepSeek-OCR Docker Container")
    print("=" * 40)
    print()
    
    # Start the API server
    print("[INFO] Starting API server on port 8000...")
    api_process = subprocess.Popen(
        [sys.executable, "/app/start_server.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # Wait for API to be ready
    print("[INFO] Waiting for API to initialize...")
    time.sleep(10)
    
    # Check if GUI should be enabled
    enable_gui = os.environ.get("ENABLE_GUI", "false").lower() == "true"
    
    gui_process = None
    if enable_gui:
        print("[INFO] GUI enabled - starting on port 7863...")
        gui_process = subprocess.Popen(
            [sys.executable, "/app/GUI.py"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"[INFO] GUI started with PID {gui_process.pid}")
    else:
        print("[INFO] GUI disabled (set ENABLE_GUI=true to enable)")
    
    print()
    print("[INFO] Services running:")
    print("  - API: http://localhost:8000")
    if enable_gui:
        print("  - GUI: http://localhost:7863")
    print()
    
    # Wait for the API server (main process)
    try:
        api_process.wait()
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
        api_process.terminate()
        if gui_process:
            gui_process.terminate()

if __name__ == "__main__":
    main()
