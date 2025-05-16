import uuid

from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status

from fleet.models import Vehicle
from shipments.models import Shipment
from assignment.models.assignment import Assignment
from assignment.models.assignment_item import AssignmentItem


class AssignmentAPITests(APITestCase):
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
            status="pending"
        )

        self.create_url = reverse("assignment-list")
        self.by_vehicle_url = lambda v_id: reverse("assignment-by-vehicle", kwargs={"vehicle_id": v_id})

    def test_create_assignment(self):
        payload = {
            "deliveries": [
                {
                    "shipment_id": self.shipment.id,
                    "location": {"lat": 7.2, "lng": 80.1},
                    "sequence": 1,
                    "load": 500,
                    "role": "pickup"
                },
                {
                    "shipment_id": self.shipment.id,
                    "location": {"lat": 7.3, "lng": 80.2},
                    "sequence": 2,
                    "load": 0,
                    "role": "delivery"
                }
            ]
        }

        response = self.client.post(self.create_url, payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Assignment.objects.count(), 1)
        self.assertEqual(AssignmentItem.objects.count(), 2)

        assignment = Assignment.objects.first()
        self.assertEqual(assignment.vehicle.vehicle_id, "TRK001")
        self.assertEqual(assignment.total_load, 500)

    def test_get_assignment_by_vehicle(self):
        assignment = Assignment.objects.create(
            vehicle=self.vehicle,
            total_load=500,
            status="created"
        )
        AssignmentItem.objects.create(
            assignment=assignment,
            shipment=self.shipment,
            delivery_sequence=1,
            delivery_location={"lat": 7.2, "lng": 80.1},
            role="pickup"
        )

        response = self.client.get(self.by_vehicle_url("TRK001"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["vehicle"], self.vehicle.vehicle_id)
        self.assertEqual(len(response.data["items"]), 1)

    def test_arrival_at_sequence_returns_correct_actions(self):
        assignment = Assignment.objects.create(
            vehicle=self.vehicle,
            total_load=500,
            status="created"
        )
        AssignmentItem.objects.create(
            assignment=assignment,
            shipment=self.shipment,
            delivery_sequence=1,
            delivery_location={"lat": 7.2, "lng": 80.1},
            role="pickup"
        )
        AssignmentItem.objects.create(
            assignment=assignment,
            shipment=self.shipment,
            delivery_sequence=2,
            delivery_location={"lat": 7.3, "lng": 80.2},
            role="delivery"
        )

        # arrive_url = reverse("assignment-arrive-sequence", kwargs={"pk": assignment.pk, "sequence": 2})
        arrive_url = f"/api/assignments/{assignment.pk}/arrive/sequence/2/"
        response = self.client.post(arrive_url, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["location"], {"lat": 7.3, "lng": 80.2})
        self.assertEqual(len(response.data["actions"]), 1)
        self.assertEqual(response.data["actions"][0]["role"], "delivery")
        self.assertEqual(response.data["actions"][0]["shipment_id"], self.shipment.id)

    def test_arrival_with_multiple_actions_at_same_location(self):
        assignment = Assignment.objects.create(
            vehicle=self.vehicle,
            total_load=800,
            status="created"
        )

        # Add another shipment
        shipment2 = Shipment.objects.create(
            shipment_id=str(uuid.uuid4()),
            order_id="ORD002",
            demand=300,
            origin={"lat": 7.1, "lng": 80.0},
            destination={"lat": 7.3, "lng": 80.2},  # SAME location as the first delivery
            status="pending"
        )

        # First shipment's delivery
        AssignmentItem.objects.create(
            assignment=assignment,
            shipment=self.shipment,
            delivery_sequence=2,
            delivery_location={"lat": 7.3, "lng": 80.2},
            role="delivery"
        )

        # Second shipment's delivery — same place
        AssignmentItem.objects.create(
            assignment=assignment,
            shipment=shipment2,
            delivery_sequence=3,
            delivery_location={"lat": 7.3, "lng": 80.2},
            role="delivery"
        )

        # Call the arrival endpoint at sequence 2 (first of the two)
        arrive_url = f"/api/assignments/{assignment.pk}/arrive/sequence/2/"
        response = self.client.post(arrive_url, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["location"], {"lat": 7.3, "lng": 80.2})
        self.assertEqual(len(response.data["actions"]), 2)

        roles = [a["role"] for a in response.data["actions"]]
        shipment_ids = [a["shipment_id"] for a in response.data["actions"]]

        self.assertIn("delivery", roles)
        self.assertIn(self.shipment.id, shipment_ids)
        self.assertIn(shipment2.id, shipment_ids)

    def test_arrival_with_invalid_sequence_returns_404(self):
        assignment = Assignment.objects.create(
            vehicle=self.vehicle,
            total_load=500,
            status="created"
        )
        # arrive_url = reverse("assignment-arrive-sequence", kwargs={"pk": assignment.pk, "sequence": 99})
        arrive_url = f"/api/assignment/assignments/{assignment.pk}/arrive/sequence/99/"
        response = self.client.post(arrive_url, format="json")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
