"""
Environment variable loading utility.

This module provides functions to load environment variables from files.
"""
import os
import logging

logger = logging.getLogger(__name__)

def load_env_from_file(file_path):
    """
    Load environment variables from a file.
    
    Args:
        file_path: Path to the environment variable file.
        
    Returns:
        True if file was loaded successfully, False otherwise.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Environment file not found: {file_path}")
        return False
        
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
                
        logger.info(f"Loaded environment variables from {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading environment variables from {file_path}: {str(e)}")
        return False
