from django.conf import settings


if settings.ENABLE_FLEET_EXTENDED_MODELS:
    from datetime import timezone
    from rest_framework import status
    from django.test import TestCase
    from rest_framework.test import APIClient

    from fleet.models import Vehicle

    class FuelRecordAPITest(TestCase):
        """Test fuel record API endpoints."""

        def setUp(self):
            self.client = APIClient()
            self.vehicle = Vehicle.objects.create(
                vehicle_id="TRK001", capacity=1000, status="available",
                fuel_type="diesel", fuel_efficiency=8.5
            )

        def test_create_fuel_record(self):
            """Test creating a new fuel record."""
            payload = {
                'vehicle': self.vehicle.id,
                'refuel_date': timezone.now().isoformat(),
                'amount': 75.5,
                'cost': 120.25,
                'odometer_reading': 5000,
                'location_name': 'Gas Station ABC'
            }
            response = self.client.post('/api/fleet/fuel/', payload, format='json')
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
            self.assertEqual(float(response.data['amount']), 75.5)
            self.assertEqual(float(response.data['cost']), 120.25)
            self.assertEqual(response.data['odometer_reading'], 5000)
