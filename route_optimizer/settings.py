import os
import sys
import logging
from pathlib import Path

# Try to load environment variables from file
try:
    from route_optimizer.utils.env_loader import load_env_from_file
    # Try different possible locations for the env file
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env_var.env'),  # App directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'env_var.env'),  # Root directory
    ]
    
    for path in env_paths:
        if load_env_from_file(path):
            break
except ImportError:
    # Module might not be available during initial imports
    pass

# Determine if we're in test mode
TESTING = 'test' in sys.argv or 'pytest' in sys.modules

# Google Maps API configuration
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not GOOGLE_MAPS_API_KEY:
    if not TESTING:
        raise ValueError("Google Maps API key is required. Set the GOOGLE_MAPS_API_KEY environment variable.")
    else:
        # Use a dummy key for testing
        GOOGLE_MAPS_API_KEY = "test_dummy_key_for_unit_tests"
        logging.warning("Using dummy Google Maps API key for testing.")

GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json'
USE_API_BY_DEFAULT = os.getenv('USE_API_BY_DEFAULT', 'False').lower() == 'true'

# API request settings
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Exponential backoff
RETRY_DELAY_SECONDS = 1
CACHE_EXPIRY_DAYS = 30

# Cache settings
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache', # For development
        'LOCATION': 'unique-snowflake',
    }
}
OPTIMIZATION_RESULT_CACHE_TIMEOUT = 3600 # 1 hour

