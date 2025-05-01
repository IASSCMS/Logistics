from django.db import models
from fleet.models import Vehicle

class Assignment(models.Model):
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    delivery_locations = models.JSONField()  # List of [longitude, latitude]
    total_load = models.PositiveIntegerField()
    optimized_distance = models.PositiveIntegerField(null=True, blank=True)  # Store distance in meters

    def __str__(self):
        return f"Assignment #{self.id} to {self.vehicle.vehicle_id}"
