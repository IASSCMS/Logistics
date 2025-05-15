import os
import django
# import pytest # pytest is often imported here for pytest specific hooks/fixtures

# Configure Django settings before any tests are run
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'route_optimizer.tests.test_settings')
django.setup()

# Example pytest fixture (if you start using pytest specific features)
# @pytest.fixture(scope='session')
# def django_db_setup(django_db_setup, django_db_blocker):
#     """Your database setup fixture"""
#     # with django_db_blocker.unblock():
#     #     # Perform any one-time database setup for the session
#     pass
