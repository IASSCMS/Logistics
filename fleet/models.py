from django.db import models
from django.core.validators import MinValueValidator
from django.utils import timezone
import uuid

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


class MaintenanceRecord(models.Model):
    """
    Model for tracking vehicle maintenance history.
    """
    MAINTENANCE_TYPE_CHOICES = [
        ('routine', 'Routine Checkup'),
        ('repair', 'Repair'),
        ('scheduled', 'Scheduled Maintenance'),
        ('emergency', 'Emergency Repair')
    ]
    
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled')
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, related_name='maintenance_records')
    maintenance_type = models.CharField(max_length=20, choices=MAINTENANCE_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')
    description = models.TextField()
    
    # Scheduling information
    scheduled_date = models.DateField()
    completion_date = models.DateField(null=True, blank=True)
    
    # Cost information
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.vehicle.vehicle_id} - {self.maintenance_type} ({self.status})"
    
    def complete_maintenance(self, completion_date=None, cost=None):
        """Mark maintenance as completed with optional completion date and cost."""
        self.status = 'completed'
        if completion_date:
            self.completion_date = completion_date
        else:
            self.completion_date = timezone.now().date()
        
        if cost is not None:
            self.cost = cost
        
        self.save(update_fields=['status', 'completion_date', 'cost', 'updated_at'])
        
        # Update vehicle status if it was in maintenance
        if self.vehicle.status == 'maintenance':
            self.vehicle.status = 'available'
            self.vehicle.save(update_fields=['status', 'updated_at'])


class FuelRecord(models.Model):
    """
    Model for tracking fuel consumption and refueling.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, related_name='fuel_records')
    
    # Refueling information
    refuel_date = models.DateTimeField()
    amount = models.DecimalField(max_digits=6, decimal_places=2, validators=[MinValueValidator(0.01)])
    cost = models.DecimalField(max_digits=8, decimal_places=2, validators=[MinValueValidator(0.01)])
    odometer_reading = models.PositiveIntegerField(help_text="Odometer reading in kilometers")
    
    # Location of refueling (optional)
    location_name = models.CharField(max_length=100, blank=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.vehicle.vehicle_id} - {self.refuel_date.strftime('%Y-%m-%d')} ({self.amount} L)"
    
    class Meta:
        ordering = ['-refuel_date']


class TripRecord(models.Model):
    """
    Model for tracking trip information and mileage.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, related_name='trip_records')
    
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    
    start_odometer = models.PositiveIntegerField()
    end_odometer = models.PositiveIntegerField(null=True, blank=True)
    
    start_latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    start_longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    end_latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    end_longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    driver_name = models.CharField(max_length=100, blank=True)
    purpose = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.vehicle.vehicle_id} - {self.start_time.strftime('%Y-%m-%d')}"
    
    @property
    def distance(self):
        """Calculate the distance traveled in kilometers."""
        if self.end_odometer and self.start_odometer:
            return self.end_odometer - self.start_odometer
        return None
    
    @property
    def duration(self):
        """Calculate the trip duration in minutes."""
        if self.end_time and self.start_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() / 60
        return None
    
    class Meta:
        ordering = ['-start_time']


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