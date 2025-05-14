from django.conf import settings

if settings.ENABLE_FLEET_EXTENDED_MODELS:
    from datetime import timezone, timedelta
    from rest_framework import status
    from django.test import TestCase
    from rest_framework.test import APIClient

    from fleet.models import Vehicle, MaintenanceRecord


    class MaintenanceAPITest(TestCase):
        """Test maintenance API endpoints."""

        def setUp(self):
            self.client = APIClient()
            self.vehicle = Vehicle.objects.create(
                vehicle_id="TRK001", capacity=1000, status="available"
            )
            self.maintenance = MaintenanceRecord.objects.create(
                vehicle=self.vehicle,
                maintenance_type="routine",
                status="scheduled",
                description="Oil change and inspection",
                scheduled_date=timezone.now().date() + timedelta(days=3)
            )

        def test_list_maintenance_records(self):
            """Test retrieving all maintenance records."""
            response = self.client.get('/api/fleet/maintenance/')
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertEqual(len(response.data), 1)

        def test_create_maintenance_record(self):
            """Test creating a new maintenance record."""
            scheduled_date = (timezone.now().date() + timedelta(days=5)).isoformat()
            payload = {
                'vehicle': self.vehicle.id,
                'maintenance_type': 'repair',
                'status': 'scheduled',
                'description': 'Brake replacement',
                'scheduled_date': scheduled_date
            }
            response = self.client.post('/api/fleet/maintenance/', payload, format='json')
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
            self.assertEqual(response.data['maintenance_type'], 'repair')
            self.assertEqual(response.data['status'], 'scheduled')

        def test_complete_maintenance(self):
            """Test completing a maintenance record."""
            completion_date = timezone.now().date().isoformat()
            payload = {
                'completion_date': completion_date,
                'cost': 250.75
            }
            response = self.client.post(
                f'/api/fleet/maintenance/{self.maintenance.id}/complete/',
                payload,
                format='json'
            )
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertEqual(response.data['status'], 'completed')
            self.assertEqual(response.data['completion_date'], completion_date)
            self.assertEqual(float(response.data['cost']), 250.75)
