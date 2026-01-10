"""
Location-specific configuration for LOCAL development.

This file is NOT copied to Docker. The Docker container has its own
version at docker/config_location.py which gets copied to Lib/.
"""

from pathlib import Path
from .config import Config

# Local development configuration
config = Config(
    # API settings (Docker container on localhost)
    api_host="localhost",
    api_port=8000,
    
    # Local directories
    upload_dir=Path("data/uploads"),
    results_dir=Path("data/results"),
    images_dir=Path("data/images"),
    
    # GUI settings
    gui_host="0.0.0.0",
    gui_port=7862,  # Local GUI port
    
    # Environment
    environment="local",
)

# Ensure directories exist
config.ensure_directories()
