from django.test import TestCase
from fleet.models import Vehicle
from django.utils import timezone


class VehicleModelTest(TestCase):
    """Unit tests for the Vehicle model."""

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

    def test_vehicle_fields_and_defaults(self):
        """Test vehicle creation and default values."""
        v = self.vehicle
        self.assertEqual(v.vehicle_id, "TRK001")
        self.assertEqual(v.name, "Test Truck 1")
        self.assertEqual(v.capacity, 1000)
        self.assertEqual(v.status, "available")
        self.assertEqual(v.fuel_type, "diesel")
        self.assertTrue(v.is_available)
        self.assertIsNone(v.current_latitude)
        self.assertIsNone(v.current_longitude)
        self.assertIsNone(v.last_location_update)

    def test_update_location_sets_values_and_timestamp(self):
        """Test updating vehicle's current location."""
        lat, lon = 45.123456, -75.654321

        self.vehicle.update_location(lat, lon)
        self.vehicle.refresh_from_db()

        self.assertEqual(float(self.vehicle.current_latitude), lat)
        self.assertEqual(float(self.vehicle.current_longitude), lon)
        self.assertIsNotNone(self.vehicle.last_location_update)

        now = timezone.now()
        self.assertLess(abs((now - self.vehicle.last_location_update).total_seconds()), 5)

    def test_location_is_stale_logic(self):
        """Test location_is_stale property."""
        # Initially: no location update → should be stale
        self.assertTrue(self.vehicle.location_is_stale)

        # After updating location → should not be stale
        self.vehicle.update_location(10.0, 20.0)
        self.assertFalse(self.vehicle.location_is_stale)
