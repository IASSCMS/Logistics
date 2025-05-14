from typing import List, Optional
from django.db import models
from dataclasses import dataclass, field
from route_optimizer.core.types_1 import Location
from route_optimizer.core.constants import DEFAULT_DELIVERY_PRIORITY, PRIORITY_NORMAL

@dataclass
class Vehicle:
    """Class representing a vehicle with capacity and other constraints."""
    id: str
    capacity: float
    start_location_id: str  # Where the vehicle starts from
    end_location_id: Optional[str] = None  # Where the vehicle must end (if different)
    cost_per_km: float = 1.0  # Cost per kilometer
    fixed_cost: float = 0.0   # Fixed cost for using this vehicle
    max_distance: Optional[float] = None  # Maximum distance the vehicle can travel
    max_stops: Optional[int] = None  # Maximum number of stops
    available: bool = True
    skills: List[str] = field(default_factory=list)  # Skills/capabilities this vehicle has


@dataclass
class Delivery:
    """Class representing a delivery with demand and constraints."""
    id: str
    location_id: str
    demand: float  # Demand quantity
    priority: int = DEFAULT_DELIVERY_PRIORITY  # = normal, higher values = higher priority
    required_skills: List[str] = field(default_factory=list)  # Required skills
    is_pickup: bool = False  # True for pickup, False for delivery

class DistanceMatrixCache(models.Model):
    """Cache for distance matrices to reduce API calls."""
    cache_key = models.CharField(max_length=255, unique=True)
    matrix_data = models.TextField()  # JSON serialized distance matrix
    location_ids = models.TextField()  # JSON serialized location IDs
    time_matrix_data = models.TextField(null=True, blank=True)  # JSON serialized time matrix
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Distance Matrix Cache"
        verbose_name_plural = "Distance Matrix Caches"
        indexes = [
            models.Index(fields=['cache_key']),
            models.Index(fields=['created_at']),
        ]
