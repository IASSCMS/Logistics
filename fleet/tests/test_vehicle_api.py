from rest_framework.test import APIClient
from django.test import TestCase
from rest_framework import status

from fleet.models import Vehicle, VehicleLocation


class VehicleAPITest(TestCase):
    """Test vehicle API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.vehicle1 = Vehicle.objects.create(
            vehicle_id="TRK001", capacity=1000, status="available",
            name="Truck 1", fuel_type="diesel"
        )
        self.vehicle2 = Vehicle.objects.create(
            vehicle_id="TRK002", capacity=500, status="maintenance",
            name="Truck 2", fuel_type="petrol"
        )
        self.vehicle3 = Vehicle.objects.create(
            vehicle_id="TRK003", capacity=750, status="assigned",
            name="Truck 3", fuel_type="diesel"
        )

    def test_list_all_vehicles(self):
        """Test retrieving all vehicles."""
        response = self.client.get('/api/fleet/vehicles/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 3)

    def test_filter_by_status(self):
        """Test filtering vehicles by status."""
        response = self.client.get('/api/fleet/vehicles/?status=available')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['vehicle_id'], 'TRK001')

    def test_filter_by_min_capacity(self):
        """Test filtering vehicles by minimum capacity."""
        response = self.client.get('/api/fleet/vehicles/?min_capacity=800')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['vehicle_id'], 'TRK001')

    def test_filter_by_fuel_type(self):
        """Test filtering vehicles by fuel type."""
        response = self.client.get('/api/fleet/vehicles/?fuel_type=diesel')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
        vehicle_ids = [v['vehicle_id'] for v in response.data]
        self.assertIn('TRK001', vehicle_ids)
        self.assertIn('TRK003', vehicle_ids)

    def test_create_vehicle(self):
        """Test creating a new vehicle."""
        payload = {
            'vehicle_id': 'TRK004',
            'name': 'Truck 4',
            'capacity': 1200,
            'status': 'available',
            'fuel_type': 'electric',
            'plate_number': 'XYZ789'
        }
        response = self.client.post('/api/fleet/vehicles/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['vehicle_id'], 'TRK004')
        self.assertEqual(response.data['fuel_type'], 'electric')

    def test_update_vehicle(self):
        """Test updating an existing vehicle."""
        response = self.client.patch(
            f'/api/fleet/vehicles/{self.vehicle1.id}/',
            {'status': 'maintenance'},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'maintenance')

    def test_update_location(self):
        """Test updating vehicle location."""
        payload = {
            'latitude': 42.123456,
            'longitude': -71.654321,
            'speed': 65.5
        }
        response = self.client.post(
            f'/api/fleet/vehicles/{self.vehicle1.id}/update_location/',
            payload,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check that location was updated in the vehicle
        self.vehicle1.refresh_from_db()
        self.assertEqual(float(self.vehicle1.current_latitude), 42.123456)
        self.assertEqual(float(self.vehicle1.current_longitude), -71.654321)

        # Check that a location history record was created
        location_history = VehicleLocation.objects.filter(vehicle=self.vehicle1)
        self.assertEqual(location_history.count(), 1)
        self.assertEqual(float(location_history[0].speed), 65.5)
