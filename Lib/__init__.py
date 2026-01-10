"""
DeepSeek-OCR Library

Provides API client, post-processing, and file management utilities for DeepSeek-OCR.
Can be used by GUI, CLI scripts, or Docker-internal applications.
"""

from .config import Config
from .config_location import config
from .ocr_client import OCRClient
from .postprocessor import PostProcessor
from .file_utils import FileManager

__all__ = [
    'Config',
    'config',
    'OCRClient', 
    'PostProcessor',
    'FileManager'
]
