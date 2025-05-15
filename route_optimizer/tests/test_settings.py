import os
from pathlib import Path

# It's generally better to define test settings explicitly or import only what's absolutely necessary
# from the main settings and then override.
# For simplicity, if your main settings are well-structured with defaults,
# you might import common non-sensitive structures.
# However, for full control in tests, defining explicitly is often clearer.

# --- Start with a clean slate or minimal necessary imports ---
# from logistics_core.settings import BASE_DIR, INSTALLED_APPS as BASE_INSTALLED_APPS # Example if needed

BASE_DIR = Path(__file__).resolve().parent.parent.parent # Assuming this file is in route_optimizer/tests/

SECRET_KEY = 'dummy-secret-key-for-testing-route-optimizer'
DEBUG = False  # Tests should generally run with DEBUG=False unless specifically testing debug features
TESTING = True # Explicitly set for tests

ALLOWED_HOSTS = ['testserver', 'localhost', '127.0.0.1'] # 'testserver' is used by Django's test client

# Use an in-memory database for faster tests
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Minimal INSTALLED_APPS for route_optimizer tests
# Add other apps ONLY IF route_optimizer has direct dependencies (e.g., ForeignKey to their models)
# that are exercised in its tests.
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework', # If your app's API tests use DRF features
    'drf_yasg',     # If schema generation is tested or needed
    'route_optimizer',
    # 'fleet',        # Example: Uncomment if route_optimizer tests interact with fleet models
    # 'assignment',   # Example: Uncomment if route_optimizer tests interact with assignment models
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

ROOT_URLCONF = 'route_optimizer.api.urls' # Point to app's API URLs for isolated view testing

# --- route_optimizer specific settings for testing ---
GOOGLE_MAPS_API_KEY = 'test-api-key-for-route-optimizer-tests'
GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json' # Or mock server URL
USE_API_BY_DEFAULT = False # Usually False for tests to avoid external calls

# API request settings for faster tests
MAX_RETRIES = 1
BACKOFF_FACTOR = 0.1
RETRY_DELAY_SECONDS = 0.1
CACHE_EXPIRY_DAYS = 1 # Short expiry for tests if caching is tested

# Cache settings for tests (usually in-memory or dummy)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache', # Or 'locmem' if testing cache behavior
        # 'LOCATION': 'route-optimizer-test-cache',
    }
}
OPTIMIZATION_RESULT_CACHE_TIMEOUT = 60 # 1 minute for tests if caching is tested

# Logging for tests (can be minimal or configured to capture test output)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True, # Often true for tests to suppress app/django noise
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO', # Or 'DEBUG' if you need detailed logs from tests
    },
    'loggers': {
        'route_optimizer': {
            'handlers': ['console'],
            'level': 'DEBUG', # More verbose for your app during tests
            'propagate': False,
        },
    }
}
