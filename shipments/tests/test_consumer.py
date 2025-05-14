from django.test import TestCase
from shipments.models import Shipment
from shipments.consumers.order_events import handle_order_created


class KafkaConsumerRobustTest(TestCase):
    def test_valid_order_event_creates_shipment(self):
        """A valid event should create a shipment."""
        event = {
            "order_id": "ORD001",
            "origin_warehouse_id": "WH1",
            "destination_warehouse_id": "WH2"
        }
        handle_order_created(event)

        shipment = Shipment.objects.get(order_id="ORD001")
        self.assertEqual(shipment.status, "pending")
        self.assertEqual(shipment.origin_warehouse_id, "WH1")
        self.assertEqual(shipment.destination_warehouse_id, "WH2")

    def test_missing_order_id_does_not_create_shipment(self):
        """Missing order_id should skip creation."""
        event = {
            "origin_warehouse_id": "WH1",
            "destination_warehouse_id": "WH2"
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)

    def test_missing_origin_does_not_create_shipment(self):
        event = {
            "order_id": "ORD002",
            "destination_warehouse_id": "WH2"
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)

    def test_missing_destination_does_not_create_shipment(self):
        event = {
            "order_id": "ORD003",
            "origin_warehouse_id": "WH1"
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)

    def test_invalid_data_type_ignored(self):
        """If order_id is not a string, the handler should not crash."""
        event = {
            "order_id": 12345,
            "origin_warehouse_id": "WH1",
            "destination_warehouse_id": "WH2"
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.filter(order_id=12345).count(), 1)

    def test_duplicate_order_id_creates_separate_shipments(self):
        """If shipment_id is random, even duplicate order_id can create multiple records."""
        event = {
            "order_id": "ORDDUP",
            "origin_warehouse_id": "WH1",
            "destination_warehouse_id": "WH2"
        }
        handle_order_created(event)
        handle_order_created(event)
        self.assertEqual(Shipment.objects.filter(order_id="ORDDUP").count(), 2)

    def test_extra_fields_are_ignored(self):
        """Extra fields in the event should not break creation."""
        event = {
            "order_id": "ORD004",
            "origin_warehouse_id": "WH1",
            "destination_warehouse_id": "WH2",
            "customer_priority": "high",
            "notes": "this is ignored"
        }
        handle_order_created(event)
        self.assertTrue(Shipment.objects.filter(order_id="ORD004").exists())

    def test_empty_event_dict(self):
        """An empty dict should be gracefully ignored."""
        handle_order_created({})
        self.assertEqual(Shipment.objects.count(), 0)

    def test_null_values(self):
        """Null values should not create shipments."""
        event = {
            "order_id": None,
            "origin_warehouse_id": None,
            "destination_warehouse_id": None,
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)
