import os
import django
from django.conf import settings

# Configure Django settings before any tests are run
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'route_optimizer.tests.test_settings')
django.setup()

# # For running all tests
# python -m pytest route_optimizer/tests/ --ds route_optimizer.tests.test_settings
# python -m pytest route_optimizer/tests/
# python -m pytest route_optimizer/tests/ --django-settings=route_optimizer.tests.test_settings

# # For Django's built-in test runner
# python manage.py test route_optimizer --settings=route_optimizer.tests.test_settings