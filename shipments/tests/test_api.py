from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from django.utils import timezone
from shipments.models import Shipment
from datetime import timedelta
from django.core.exceptions import ValidationError

class ShipmentAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.shipment = Shipment.objects.create(
            shipment_id="SHIP123",
            order_id="ORD456",
            origin_warehouse_id="WH001",
            destination_warehouse_id="WH002",
        )

    def test_create_shipment(self):
        payload = {
            "shipment_id": "SHIP999",
            "order_id": "ORD999",
            "origin_warehouse_id": "WH010",
            "destination_warehouse_id": "WH020"
        }
        response = self.client.post("/api/shipments/", payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["shipment_id"], "SHIP999")

    def test_mark_scheduled(self):
        scheduled_time = (timezone.now() + timedelta(days=1)).isoformat()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_scheduled/", {
            "scheduled_time": scheduled_time
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "scheduled")

    def test_mark_dispatched(self):
        self.shipment.mark_scheduled(timezone.now())
        dispatch_time = (timezone.now() + timedelta(hours=1)).isoformat()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_dispatched/", {
            "dispatch_time": dispatch_time
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "dispatched")

    def test_mark_in_transit(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_in_transit/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "in_transit")

    def test_mark_delivered(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        self.shipment.mark_in_transit()
        delivery_time = timezone.now().isoformat()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_delivered/", {
            "delivery_time": delivery_time
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "delivered")

    def test_mark_failed(self):
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_failed/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "failed")

    def test_invalid_transition_dispatched_without_schedule(self):
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_dispatched/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_invalid_transition_delivered_without_in_transit(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_delivered/", {
            "delivery_time": timezone.now().isoformat()
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_invalid_transition_failed_after_delivery(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        self.shipment.mark_in_transit()
        self.shipment.mark_delivered()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_failed/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_revert_to_pending_from_scheduled(self):
        self.shipment.mark_scheduled()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_pending/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "pending")

    def test_invalid_revert_to_pending_from_dispatched(self):
        self.shipment.mark_scheduled()
        self.shipment.mark_dispatched()
        response = self.client.post(f"/api/shipments/{self.shipment.id}/mark_pending/", {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_duplicate_shipment_id(self):
        with self.assertRaises(Exception):
            Shipment.objects.create(
                shipment_id="SHIP123",
                order_id="ORD999",
                origin_warehouse_id="WHX",
                destination_warehouse_id="WHY"
            )
