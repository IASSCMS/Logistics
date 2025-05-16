import uuid
from django.utils import timezone
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status

from fleet.models import Vehicle
from shipments.models import Shipment
from assignment.models.assignment import Assignment
from assignment.models.assignment_item import AssignmentItem


class AssignmentActionCompletionTests(APITestCase):
    def setUp(self):
        self.vehicle = Vehicle.objects.create(
            vehicle_id="TRK001",
            name="Truck 1",
            capacity=1000,
            status="available",
            fuel_type="diesel"
        )

        self.shipment = Shipment.objects.create(
            shipment_id=str(uuid.uuid4()),
            order_id="ORD001",
            demand=500,
            origin={"lat": 7.2, "lng": 80.1},
            destination={"lat": 7.3, "lng": 80.2},
            status="in_transit"
        )

        self.assignment = Assignment.objects.create(
            vehicle=self.vehicle,
            total_load=500,
            status="created"
        )

        self.pickup_item = AssignmentItem.objects.create(
            assignment=self.assignment,
            shipment=self.shipment,
            delivery_sequence=1,
            delivery_location=self.shipment.origin,
            role="pickup",
            is_delivered=False
        )

        self.delivery_item = AssignmentItem.objects.create(
            assignment=self.assignment,
            shipment=self.shipment,
            delivery_sequence=2,
            delivery_location=self.shipment.destination,
            role="delivery",
            is_delivered=False
        )

    def test_confirm_delivery_action_successfully(self):
        url = f"/api/assignments/{self.assignment.id}/actions/{self.delivery_item.id}/complete/"
        response = self.client.post(url, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], "Delivery confirmed")
        self.assertEqual(response.data["shipment_id"], self.shipment.id)
        self.assertEqual(response.data["new_status"], "delivered")

    def test_confirm_pickup_action_successfully(self):
        self.shipment.status = "scheduled"
        self.shipment.save()

        url = f"/api/assignments/{self.assignment.id}/actions/{self.pickup_item.id}/complete/"
        response = self.client.post(url, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], "Pickup confirmed")
        self.assertEqual(response.data["shipment_id"], self.shipment.id)
        self.assertEqual(response.data["new_status"], "in_transit")

    def test_confirm_action_invalid_assignment_item(self):
        url = f"/api/assignments/{self.assignment.id}/actions/9999/complete/"
        response = self.client.post(url, format="json")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_confirm_already_completed_action(self):
        self.delivery_item.is_delivered = True
        self.delivery_item.delivered_at = timezone.now()
        self.delivery_item.save()

        url = f"/api/assignments/{self.assignment.id}/actions/{self.delivery_item.id}/complete/"
        response = self.client.post(url, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], "Already marked complete")
