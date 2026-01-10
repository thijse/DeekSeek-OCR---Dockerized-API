"""
Location-specific configuration for DOCKER container.

This file is copied to /app/Lib/config_location.py during Docker build,
replacing the local development version.
"""

from pathlib import Path
from .config import Config

# Docker internal configuration
config = Config(
    # API settings (same container, localhost)
    api_host="localhost",
    api_port=8000,
    
    # Docker directories (absolute paths)
    upload_dir=Path("/app/data/uploads"),
    results_dir=Path("/app/data/results"),
    images_dir=Path("/app/data/images"),
    
    # GUI settings
    gui_host="0.0.0.0",
    gui_port=7863,  # Different port than local to avoid conflicts
    
    # Environment
    environment="docker",
)

# Ensure directories exist
config.ensure_directories()
