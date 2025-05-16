from django.db import models
from fleet.models import Vehicle

class Assignment(models.Model):
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    status = models.CharField(
        max_length=32,
        choices=[
            ('created', 'Created'),
            ('dispatched', 'Dispatched'),
            ('partially_completed', 'Partially Completed'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('reassigned', 'Reassigned'),
        ],
        default='created'
    )

    total_load = models.PositiveIntegerField()

    def __str__(self):
        return f"Assignment #{self.id} to Vehicle {self.vehicle.vehicle_id}"
