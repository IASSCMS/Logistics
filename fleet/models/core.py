from django.db import models
from django.utils import timezone

class Vehicle(models.Model):
    """
    Model representing a vehicle in the fleet.
    """
    STATUS_CHOICES = [
        ('available', 'Available'),
        ('assigned', 'Assigned'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ]

    FUEL_TYPE_CHOICES = [
        ('diesel', 'Diesel'),
        ('petrol', 'Petrol'),
        ('electric', 'Electric'),
        ('hybrid', 'Hybrid'),
        ('cng', 'CNG')
    ]

    vehicle_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100, blank=True)
    capacity = models.PositiveIntegerField(help_text="Capacity in kilograms")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='available')
    fuel_type = models.CharField(max_length=20, choices=FUEL_TYPE_CHOICES, default='diesel')
    plate_number = models.CharField(max_length=20, blank=True)
    year_of_manufacture = models.PositiveIntegerField(null=True, blank=True)

    # Depot
    depot_id = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text="External ID of the depot this vehicle is assigned to"
    )
    depot_latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    depot_longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    # Location tracking
    current_latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    current_longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    last_location_update = models.DateTimeField(null=True, blank=True)

    # Additional specifications
    max_speed = models.PositiveIntegerField(null=True, blank=True, help_text="Maximum speed in km/h")
    fuel_efficiency = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Fuel efficiency in km/l"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.vehicle_id} ({self.status})"

    def update_location(self, latitude, longitude):
        """Update the vehicle's current location and timestamp."""
        self.current_latitude = latitude
        self.current_longitude = longitude
        self.last_location_update = timezone.now()
        self.save(update_fields=['current_latitude', 'current_longitude', 'last_location_update'])

    @property
    def is_available(self):
        """Check if vehicle is available for assignment."""
        return self.status == 'available'

    @property
    def location_is_stale(self):
        """Check if location data is stale (more than 30 minutes old)."""
        if not self.last_location_update:
            return True
        return (timezone.now() - self.last_location_update).total_seconds() > 1800  # 30 minutes

    class Meta:
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['vehicle_id']),
        ]

class VehicleLocation(models.Model):
    """
    Model for tracking historical vehicle locations.
    """
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, related_name='location_history')
    timestamp = models.DateTimeField(auto_now_add=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    speed = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, help_text="Speed in km/h")
    heading = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, help_text="Heading in degrees")

    def __str__(self):
        return f"{self.vehicle.vehicle_id} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = "Vehicle Locations"
