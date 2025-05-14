from django.test import TestCase
from rest_framework.test import APIClient
from fleet.models import Vehicle
from assignment.models import Assignment

class AssignmentAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.vehicle = Vehicle.objects.create(vehicle_id="TRK001", capacity=100, status="available")
        self.busy_vehicle = Vehicle.objects.create(vehicle_id="TRK002", capacity=80, status="assigned")

    def test_create_assignment_success(self):
        payload = {
            "deliveries": [
                {"location": [77.59, 12.97], "load": 40},
                {"location": [77.61, 12.98], "load": 30}
            ]
        }
        response = self.client.post('/api/assignment/assignments/', payload, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['total_load'], 70)

    def test_create_assignment_insufficient_capacity(self):
        payload = {
            "deliveries": [{"location": [77.59, 12.97], "load": 150}]
        }
        response = self.client.post('/api/assignment/assignments/', payload, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("No available vehicle", response.data['error'])

    def test_create_assignment_with_no_available_vehicle(self):
        self.vehicle.status = "maintenance"
        self.vehicle.save()
        payload = {"deliveries": [{"location": [77.59, 12.97], "load": 50}]}
        response = self.client.post('/api/assignment/assignments/', payload, format='json')
        self.assertEqual(response.status_code, 400)

    def test_create_assignment_with_no_deliveries(self):
        payload = {}
        response = self.client.post('/api/assignment/assignments/', payload, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("Deliveries required", response.data['error'])

    def test_get_all_assignments(self):
        Assignment.objects.create(
            vehicle=self.vehicle,
            delivery_locations=[[77.59, 12.97]],
            total_load=50
        )
        response = self.client.get('/api/assignment/assignments/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_vehicle_marked_assigned_after_assignment(self):
        payload = {
            "deliveries": [{"location": [77.59, 12.97], "load": 50}]
        }
        self.client.post('/api/assignment/assignments/', payload, format='json')
        self.vehicle.refresh_from_db()
        self.assertEqual(self.vehicle.status, "assigned")

    def test_assignment_model_str(self):
        assignment = Assignment.objects.create(
            vehicle=self.vehicle,
            delivery_locations=[[77.59, 12.97]],
            total_load=50
        )
        expected = f"Assignment #{assignment.id} to {self.vehicle.vehicle_id}"
        self.assertEqual(str(assignment), expected)
