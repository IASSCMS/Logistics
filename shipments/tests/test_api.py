from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from django.utils import timezone
from datetime import timedelta
from shipments.models import Shipment


class ShipmentAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.shipment = Shipment.objects.create(
            shipment_id="SHIP123",
            order_id="ORD456",
            origin={"lat": 6.9271, "lng": 79.8612},
            destination={"lat": 7.2906, "lng": 80.6337},
            demand=50,
        )

    def create_shipment(self, shipment_id="SHIP999", demand=75):
        payload = {
            "shipment_id": shipment_id,
            "order_id": "ORD999",
            "origin": {"lat": 6.9, "lng": 79.8},
            "destination": {"lat": 7.2, "lng": 80.6},
            "demand": demand,
        }
        return self.client.post("/api/shipments/", payload, format="json")

    def test_create_shipment(self):
        response = self.create_shipment()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED, msg=response.data)
        self.assertEqual(response.data["shipment_id"], "SHIP999")
        self.assertEqual(response.data["demand"], 75)

    def test_mark_scheduled(self):
        scheduled_time = (timezone.now() + timedelta(days=1)).isoformat()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_scheduled/", {
            "scheduled_time": scheduled_time
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK, msg=response.data)
        self.assertEqual(response.data["status"], "scheduled")

    def test_mark_dispatched(self):
        self.shipment.mark_scheduled(timezone.now())
        dispatch_time = (timezone.now() + timedelta(hours=1)).isoformat()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_dispatched/", {
            "dispatch_time": dispatch_time
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK, msg=response.data)
        self.assertEqual(response.data["status"], "dispatched")

    def test_mark_in_transit(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_in_transit/")
        self.assertEqual(response.status_code, status.HTTP_200_OK, msg=response.data)
        self.assertEqual(response.data["status"], "in_transit")

    def test_mark_delivered(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        self.shipment.mark_in_transit()
        delivery_time = timezone.now().isoformat()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_delivered/", {
            "delivery_time": delivery_time
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK, msg=response.data)
        self.assertEqual(response.data["status"], "delivered")

    def test_mark_failed(self):
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_failed/")
        self.assertEqual(response.status_code, status.HTTP_200_OK, msg=response.data)
        self.assertEqual(response.data["status"], "failed")

    def test_invalid_transition_dispatched_without_schedule(self):
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_dispatched/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST, msg=response.data)
        self.assertIn("error", response.data)

    def test_invalid_transition_delivered_without_in_transit(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_delivered/", {
            "delivery_time": timezone.now().isoformat()
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST, msg=response.data)
        self.assertIn("error", response.data)

    def test_invalid_transition_failed_after_delivery(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        self.shipment.mark_in_transit()
        self.shipment.mark_delivered()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_failed/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST, msg=response.data)
        self.assertIn("error", response.data)

    def test_revert_to_pending_from_scheduled(self):
        self.shipment.mark_scheduled()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_pending/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK, msg=response.data)
        self.assertEqual(response.data["status"], "pending")

    def test_invalid_revert_to_pending_from_dispatched(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_pending/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST, msg=response.data)
        self.assertIn("error", response.data)

    def test_duplicate_shipment_id(self):
        with self.assertRaises(Exception):
            Shipment.objects.create(
                shipment_id="SHIP123",
                order_id="ORD999",
                origin={"lat": 1.0, "lng": 2.0},
                destination={"lat": 3.0, "lng": 4.0},
                demand=20,
            )
