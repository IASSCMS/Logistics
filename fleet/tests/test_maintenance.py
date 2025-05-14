from datetime import timezone, timedelta

from django.conf import settings
if settings.ENABLE_FLEET_EXTENDED_MODELS:
    from django.test import TestCase
    from fleet.models import Vehicle, MaintenanceRecord

    class MaintenanceRecordModelTest(TestCase):
        """Test maintenance record functionality."""

        def setUp(self):
            self.vehicle = Vehicle.objects.create(
                vehicle_id="TRK002",
                capacity=1500,
                status="available"
            )

            self.maintenance = MaintenanceRecord.objects.create(
                vehicle=self.vehicle,
                maintenance_type="routine",
                status="scheduled",
                description="Regular oil change",
                scheduled_date=timezone.now().date() + timedelta(days=5)
            )

        def test_complete_maintenance(self):
            """Test completing maintenance."""
            # Set vehicle to maintenance status
            self.vehicle.status = "maintenance"
            self.vehicle.save()

            # Complete maintenance
            completion_date = timezone.now().date()
            cost = 150.75

            self.maintenance.complete_maintenance(completion_date, cost)

            # Check that maintenance is completed
            self.maintenance.refresh_from_db()
            self.assertEqual(self.maintenance.status, "completed")
            self.assertEqual(self.maintenance.completion_date, completion_date)
            self.assertEqual(float(self.maintenance.cost), cost)

            # Check that vehicle status was updated
            self.vehicle.refresh_from_db()
            self.assertEqual(self.vehicle.status, "available")

