"""
Utilities for persisting data to disk and loading it on startup.
"""

import json
import os
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 5 * 1024 * 1024

def save_to_json(data: Dict[str, Any], filename: str) -> bool:
    """
    Save dictionary data to a JSON file.

    Args:
        data: The dictionary to save
        filename: The filename to save to

    Returns:
        True if successful, False otherwise
    """
    try:
        tmp = f"{filename}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, filename)  # atomic: no half-written file on crash
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {str(e)}")
        return False

def load_from_json(filename: str) -> Dict[str, Any]:
    """
    Load dictionary data from a JSON file.

    Args:
        filename: The filename to load from

    Returns:
        The loaded dictionary, or an empty dictionary if the file doesn't exist
    """
    if not os.path.exists(filename):
        return {}

    try:
        if os.path.getsize(filename) > MAX_FILE_BYTES:
            logger.error(f"Refusing to load {filename}: larger than {MAX_FILE_BYTES} bytes")
            return {}
        with open(filename, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error(f"Refusing to load {filename}: top level is not an object")
            return {}
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        return {}
