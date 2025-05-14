from django.conf import settings

if settings.ENABLE_FLEET_EXTENDED_MODELS:
    from datetime import timezone, timedelta
    from rest_framework import status
    from django.test import TestCase
    from rest_framework.test import APIClient

    from fleet.models import Vehicle, TripRecord


    class TripRecordAPITest(TestCase):
        """Test trip record API endpoints."""

        def setUp(self):
            self.client = APIClient()
            self.vehicle = Vehicle.objects.create(
                vehicle_id="TRK001", capacity=1000, status="available"
            )

            start_time = timezone.now() - timedelta(hours=2)
            self.trip = TripRecord.objects.create(
                vehicle=self.vehicle,
                start_time=start_time,
                start_odometer=5000,
                driver_name="Test Driver",
                purpose="Delivery to Warehouse A"
            )

        def test_create_trip_record(self):
            """Test creating a new trip record."""
            start_time = (timezone.now() - timedelta(hours=1)).isoformat()
            payload = {
                'vehicle': self.vehicle.id,
                'start_time': start_time,
                'start_odometer': 5500,
                'driver_name': 'Another Driver',
                'purpose': 'Pickup from Supplier B'
            }
            response = self.client.post('/api/fleet/trips/', payload, format='json')
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
            self.assertEqual(response.data['start_odometer'], 5500)
            self.assertEqual(response.data['driver_name'], 'Another Driver')

        def test_end_trip(self):
            """Test ending a trip."""
            end_time = timezone.now().isoformat()
            payload = {
                'end_time': end_time,
                'end_odometer': 5150,
                'end_latitude': 40.123456,
                'end_longitude': -74.654321,
                'notes': 'Trip completed successfully'
            }
            response = self.client.post(
                f'/api/fleet/trips/{self.trip.id}/end_trip/',
                payload,
                format='json'
            )
            self.assertEqual(response.status_code, status.HTTP_200_OK)

            # Check that trip was updated
            self.trip.refresh_from_db()
            self.assertEqual(self.trip.end_odometer, 5150)
            self.assertEqual(float(self.trip.end_latitude), 40.123456)
            self.assertEqual(float(self.trip.end_longitude), -74.654321)

            # Verify calculated properties
            self.assertEqual(self.trip.distance, 150)  # 5150 - 5000
