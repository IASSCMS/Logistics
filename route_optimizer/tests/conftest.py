import os
import django
import pytest

# Configure Django settings before any tests are run
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'route_optimizer.tests.test_settings')
django.setup()

@pytest.fixture(scope='session')
def django_db_setup():
    """Fixture to set up the test database"""
    pass
