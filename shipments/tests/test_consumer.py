from django.test import TestCase
from shipments.models import Shipment
from shipments.consumers.order_events import handle_order_created


class KafkaConsumerRobustTest(TestCase):
    def test_valid_order_event_creates_shipment(self):
        event = {
            "order_id": "ORD001",
            "origin": {"lat": 6.9271, "lng": 79.8612},
            "destination": {"lat": 7.2906, "lng": 80.6337},
            "demand": 25
        }
        handle_order_created(event)

        shipment = Shipment.objects.get(order_id="ORD001")
        self.assertEqual(shipment.status, "pending")
        self.assertEqual(shipment.origin, event["origin"])
        self.assertEqual(shipment.destination, event["destination"])
        self.assertEqual(shipment.demand, 25)

    def test_missing_order_id_does_not_create_shipment(self):
        event = {
            "origin": {"lat": 6.9271, "lng": 79.8612},
            "destination": {"lat": 7.2906, "lng": 80.6337},
            "demand": 10
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)

    def test_missing_origin_does_not_create_shipment(self):
        event = {
            "order_id": "ORD002",
            "destination": {"lat": 7.2906, "lng": 80.6337},
            "demand": 10
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)

    def test_missing_destination_does_not_create_shipment(self):
        event = {
            "order_id": "ORD003",
            "origin": {"lat": 6.9271, "lng": 79.8612},
            "demand": 10
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)

    def test_invalid_data_type_for_order_id_is_casted(self):
        event = {
            "order_id": 12345,
            "origin": {"lat": 6.9, "lng": 79.8},
            "destination": {"lat": 7.3, "lng": 80.6},
            "demand": 40
        }
        handle_order_created(event)
        self.assertTrue(Shipment.objects.filter(order_id=str(12345)).exists())

    def test_duplicate_order_id_creates_multiple_shipments(self):
        event = {
            "order_id": "ORDDUP",
            "origin": {"lat": 6.9, "lng": 79.8},
            "destination": {"lat": 7.3, "lng": 80.6},
            "demand": 50
        }
        handle_order_created(event)
        handle_order_created(event)
        self.assertEqual(Shipment.objects.filter(order_id="ORDDUP").count(), 2)

    def test_extra_fields_are_ignored_and_demand_saved(self):
        event = {
            "order_id": "ORD004",
            "origin": {"lat": 6.9, "lng": 79.8},
            "destination": {"lat": 7.3, "lng": 80.6},
            "customer_priority": "high",
            "notes": "this should be ignored",
            "demand": 60
        }
        handle_order_created(event)
        shipment = Shipment.objects.get(order_id="ORD004")
        self.assertEqual(shipment.demand, 60)

    def test_event_with_no_fields_does_nothing(self):
        handle_order_created({})
        self.assertEqual(Shipment.objects.count(), 0)

    def test_null_values_are_ignored(self):
        event = {
            "order_id": None,
            "origin": None,
            "destination": None,
            "demand": None
        }
        handle_order_created(event)
        self.assertEqual(Shipment.objects.count(), 0)

    def test_negative_demand_defaults_to_zero(self):
        event = {
            "order_id": "ORD_NEG",
            "origin": {"lat": 6.9, "lng": 79.8},
            "destination": {"lat": 7.3, "lng": 80.6},
            "demand": -5
        }
        handle_order_created(event)
        shipment = Shipment.objects.get(order_id="ORD_NEG")
        self.assertEqual(shipment.demand, 0)

    def test_missing_demand_defaults_to_zero(self):
        event = {
            "order_id": "ORD_NO_DEMAND",
            "origin": {"lat": 6.9, "lng": 79.8},
            "destination": {"lat": 7.3, "lng": 80.6}
        }
        handle_order_created(event)
        shipment = Shipment.objects.get(order_id="ORD_NO_DEMAND")
        self.assertEqual(shipment.demand, 0)
