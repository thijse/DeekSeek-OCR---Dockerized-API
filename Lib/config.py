"""
Configuration management for DeepSeek-OCR.

Supports multiple environments:
- Local development (GUI.py running on host)
- Docker internal (GUI running inside Docker container)
- CLI scripts
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for DeepSeek-OCR client and processing."""
    
    # API settings
    api_host: str = "localhost"
    api_port: int = 8000
    api_timeout_create: int = 120  # Timeout for job creation
    api_timeout_status: int = 10   # Timeout for status checks
    api_timeout_download: int = 60 # Timeout for result download
    poll_interval: int = 2         # Seconds between status polls
    
    # Directory settings
    upload_dir: Path = field(default_factory=lambda: Path("data/uploads"))
    results_dir: Path = field(default_factory=lambda: Path("data/results"))
    images_dir: Path = field(default_factory=lambda: Path("data/images"))
    
    # GUI settings
    gui_host: str = "0.0.0.0"
    gui_port: int = 7862
    
    # Processing settings
    default_dpi: int = 144
    
    # Environment identifier
    environment: str = "local"  # 'local' or 'docker'
    
    @property
    def api_base_url(self) -> str:
        """Get the full API base URL."""
        return f"http://{self.api_host}:{self.api_port}"
    
    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
