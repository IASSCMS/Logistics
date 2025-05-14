from django.test import TestCase
from rest_framework.test import APIClient

from fleet.models import Vehicle


class VehicleModelTest(TestCase):
    """Test vehicle model functionality."""

    def setUp(self):
        self.vehicle = Vehicle.objects.create(
            vehicle_id="TRK001",
            name="Test Truck 1",
            capacity=1000,
            status="available",
            fuel_type="diesel",
            plate_number="ABC123",
            year_of_manufacture=2020,
            fuel_efficiency=8.5
        )

    def test_vehicle_creation(self):
        """Test that vehicle can be created."""
        self.assertEqual(self.vehicle.vehicle_id, "TRK001")
        self.assertEqual(self.vehicle.capacity, 1000)
        self.assertEqual(self.vehicle.status, "available")
        self.assertTrue(self.vehicle.is_available)

    def test_update_location(self):
        """Test updating vehicle location."""
        # Initial location should be None
        self.assertIsNone(self.vehicle.current_latitude)
        self.assertIsNone(self.vehicle.current_longitude)

        # Update location
        latitude = 45.123456
        longitude = -75.654321

        self.vehicle.update_location(latitude, longitude)

        # Check that location was updated
        self.assertEqual(float(self.vehicle.current_latitude), latitude)
        self.assertEqual(float(self.vehicle.current_longitude), longitude)
        self.assertIsNotNone(self.vehicle.last_location_update)

        # Check that location isn't stale right after update
        self.assertFalse(self.vehicle.location_is_stale)
