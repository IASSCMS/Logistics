# route_optimizer/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Maps API settings
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY') # Include your Google Maps API key here
if not GOOGLE_MAPS_API_KEY:
    raise ValueError("Google Maps API key is required. Set the GOOGLE_MAPS_API_KEY environment variable.")
GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json'

# API request settings
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Exponential backoff
RETRY_DELAY_SECONDS = 1
CACHE_EXPIRY_DAYS = 30

# Feature flags
USE_API_BY_DEFAULT = os.getenv('USE_API_BY_DEFAULT', 'False') == 'True'
