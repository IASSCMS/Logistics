import json
import logging
from django.test import TestCase
from shipments.models import Shipment
from confluent_kafka import Producer
from shipments.consumers.order_events import run_consumer_once
from django.conf import settings

logger = logging.getLogger(__name__)

class KafkaE2ETest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.producer = Producer({'bootstrap.servers': settings.KAFKA_BROKER_URL})

    def test_order_event_creates_shipment(self):
        order_id = "KAFKA_E2E_01"
        event = {
            "order_id": order_id,
            "origin": {"lat": 6.9271, "lng": 79.8612},
            "destination": {"lat": 7.2906, "lng": 80.6337}
        }

        # Send Kafka message
        self.producer.produce('orders.created', json.dumps(event).encode('utf-8'))
        self.producer.flush()

        # Process one message directly in test DB context
        run_consumer_once()

        # Now assert
        shipment = Shipment.objects.filter(order_id=order_id).first()
        logger.debug("Shipment: %s", shipment)
        self.assertIsNotNone(shipment, f"Shipment for {order_id} should exist")
        self.assertEqual(shipment.origin, event["origin"])
        self.assertEqual(shipment.destination, event["destination"])
        self.assertEqual(shipment.status, "pending")
