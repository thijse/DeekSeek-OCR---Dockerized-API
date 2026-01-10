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
from typing import Optional


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
    
    # Processing settings
    default_dpi: int = 144
    
    @property
    def api_base_url(self) -> str:
        """Get the full API base URL."""
        return f"http://{self.api_host}:{self.api_port}"
    
    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_environment(cls) -> 'Config':
        """
        Create config from environment variables.
        
        Environment variables:
            OCR_API_HOST: API host (default: localhost)
            OCR_API_PORT: API port (default: 8000)
            OCR_UPLOAD_DIR: Upload directory (default: data/uploads)
            OCR_RESULTS_DIR: Results directory (default: data/results)
            OCR_IMAGES_DIR: Images directory (default: data/images)
            OCR_POLL_INTERVAL: Poll interval in seconds (default: 2)
        """
        return cls(
            api_host=os.environ.get('OCR_API_HOST', 'localhost'),
            api_port=int(os.environ.get('OCR_API_PORT', '8000')),
            upload_dir=Path(os.environ.get('OCR_UPLOAD_DIR', 'data/uploads')),
            results_dir=Path(os.environ.get('OCR_RESULTS_DIR', 'data/results')),
            images_dir=Path(os.environ.get('OCR_IMAGES_DIR', 'data/images')),
            poll_interval=int(os.environ.get('OCR_POLL_INTERVAL', '2')),
        )
    
    @classmethod
    def for_docker_internal(cls) -> 'Config':
        """
        Create config for running inside Docker container.
        Uses localhost since both GUI and API are in the same container.
        """
        return cls(
            api_host="localhost",
            api_port=8000,
            upload_dir=Path("/app/data/uploads"),
            results_dir=Path("/app/data/results"),
            images_dir=Path("/app/data/images"),
        )
    
    @classmethod
    def for_local_development(cls) -> 'Config':
        """Create config for local development (default)."""
        return cls()


# Global config instance - can be overridden
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    Creates from environment if not already set.
    """
    global _config
    if _config is None:
        _config = Config.from_environment()
        _config.ensure_directories()
    return _config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _config
    _config = config
    _config.ensure_directories()
