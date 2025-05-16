import os
import django
import json
from confluent_kafka import Producer

# Setup Django (assuming this file is at the project root level)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'logistics_core.settings')
django.setup()

from django.conf import settings

# Fallback if env not set
bootstrap_servers = getattr(settings, 'KAFKA_BROKER_URL', 'localhost:9092')

producer = Producer({'bootstrap.servers': bootstrap_servers})

event = {
    "order_id": "ORD001",
    "origin": {"lat": 6.9271, "lng": 79.8612},
    "destination": {"lat": 7.2906, "lng": 80.6337},
    "demand": 25
}

producer.produce('orders.created', json.dumps(event).encode('utf-8'))
producer.flush()

print("✅ Published mock order event to 'orders.created'")
