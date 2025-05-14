import json
import logging
from django.test import TestCase
from shipments.models import Shipment
from confluent_kafka import Producer

from shipments.consumers.order_events import run_consumer_once

logger = logging.getLogger(__name__)

class KafkaE2ETest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.producer = Producer({'bootstrap.servers': 'localhost:9092'})

    def test_order_event_creates_shipment(self):
        order_id = "KAFKA_E2E_01"
        event = {
            "order_id": order_id,
            "origin_warehouse_id": "WH-X",
            "destination_warehouse_id": "WH-Y"
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
