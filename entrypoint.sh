#!/bin/sh

echo "Running makemigrations..."
python manage.py makemigrations --noinput

echo "Running migrate..."
python manage.py migrate --noinput

echo "Starting Django server on port ${DJANGO_PORT}..."
python manage.py runserver 0.0.0.0:${DJANGO_PORT}
