from rest_framework.test import APIClient
from django.test import TestCase
from rest_framework import status
from fleet.models import Vehicle, VehicleLocation


class VehicleAPITest(TestCase):
    """Integration tests for Vehicle API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.vehicle1 = Vehicle.objects.create(
            vehicle_id="TRK001", name="Truck 1", capacity=1000,
            status="available", fuel_type="diesel"
        )
        self.vehicle2 = Vehicle.objects.create(
            vehicle_id="TRK002", name="Truck 2", capacity=500,
            status="maintenance", fuel_type="petrol"
        )
        self.vehicle3 = Vehicle.objects.create(
            vehicle_id="TRK003", name="Truck 3", capacity=750,
            status="assigned", fuel_type="diesel"
        )

    def test_get_all_vehicles(self):
        """GET /api/fleet/vehicles/ should return all vehicles."""
        response = self.client.get('/api/fleet/vehicles/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 3)

    def test_filter_vehicles_by_status(self):
        """GET /api/fleet/vehicles/?status=available should return only available vehicles."""
        response = self.client.get('/api/fleet/vehicles/', {'status': 'available'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['vehicle_id'], "TRK001")

    def test_filter_vehicles_by_min_capacity(self):
        """GET /api/fleet/vehicles/?min_capacity=800 should return vehicles with capacity >= 800."""
        response = self.client.get('/api/fleet/vehicles/', {'min_capacity': 800})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['vehicle_id'], "TRK001")

    def test_filter_vehicles_by_fuel_type(self):
        """GET /api/fleet/vehicles/?fuel_type=diesel should return vehicles with diesel fuel."""
        response = self.client.get('/api/fleet/vehicles/', {'fuel_type': 'diesel'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
        vehicle_ids = [v['vehicle_id'] for v in response.data]
        self.assertIn("TRK001", vehicle_ids)
        self.assertIn("TRK003", vehicle_ids)

    def test_create_vehicle_successfully(self):
        """POST /api/fleet/vehicles/ should create a new vehicle."""
        payload = {
            "vehicle_id": "TRK004",
            "name": "Truck 4",
            "capacity": 1200,
            "status": "available",
            "fuel_type": "electric",
            "plate_number": "XYZ789"
        }
        response = self.client.post('/api/fleet/vehicles/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["vehicle_id"], "TRK004")
        self.assertEqual(response.data["fuel_type"], "electric")

    def test_patch_update_vehicle_status(self):
        """PATCH /api/fleet/vehicles/{id}/ should update vehicle status."""
        response = self.client.patch(
            f'/api/fleet/vehicles/{self.vehicle1.id}/',
            {"status": "maintenance"},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "maintenance")

    def test_update_vehicle_location(self):
        """POST /api/fleet/vehicles/{id}/update_location/ should update location and create history."""
        payload = {
            "latitude": 42.123456,
            "longitude": -71.654321,
            "speed": 65.5
        }
        response = self.client.post(
            f'/api/fleet/vehicles/{self.vehicle1.id}/update_location/',
            payload,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.vehicle1.refresh_from_db()
        self.assertAlmostEqual(float(self.vehicle1.current_latitude), 42.123456)
        self.assertAlmostEqual(float(self.vehicle1.current_longitude), -71.654321)

        # Verify historical tracking
        history = VehicleLocation.objects.filter(vehicle=self.vehicle1)
        self.assertEqual(history.count(), 1)
        self.assertAlmostEqual(float(history[0].speed), 65.5)
        self.assertAlmostEqual(float(history[0].latitude), 42.123456)

    def test_list_all_vehicles(self):
        response = self.client.get("/api/fleet/vehicles/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 3)

    def test_filter_by_status(self):
        response = self.client.get("/api/fleet/vehicles/?status=assigned")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['vehicle_id'], "TRK003")

    def test_ordering_by_updated_at(self):
        response = self.client.get("/api/fleet/vehicles/?ordering=-updated_at")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(len(response.data) >= 1)
        self.assertIn('updated_at', response.data[0])

    def test_mark_vehicle_available(self):
        response = self.client.post(f"/api/fleet/vehicles/{self.vehicle3.id}/mark_available/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.vehicle3.refresh_from_db()
        self.assertEqual(self.vehicle3.status, 'available')

    def test_mark_vehicle_assigned(self):
        response = self.client.post(f"/api/fleet/vehicles/{self.vehicle1.id}/mark_assigned/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.vehicle1.refresh_from_db()
        self.assertEqual(self.vehicle1.status, 'assigned')

    def test_change_status_to_available(self):
        response = self.client.post(f"/api/fleet/vehicles/{self.vehicle2.id}/change_status/", {
            "status": "available"
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.vehicle2.refresh_from_db()
        self.assertEqual(self.vehicle2.status, "available")

    def test_change_status_invalid(self):
        response = self.client.post(f"/api/fleet/vehicles/{self.vehicle1.id}/change_status/", {
            "status": "nonexistent"
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
