"""
Core data types for the route optimizer.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging # Add logging import

logger = logging.getLogger(__name__)

@dataclass
class Location:
    """
    Represents a geographic location with latitude and longitude.
    """
    id: str
    latitude: float
    longitude: float
    name: Optional[str] = None
    address: Optional[str] = None
    is_depot: bool = False
    time_window_start: Optional[int] = None  # In minutes from midnight
    time_window_end: Optional[int] = None    # In minutes from midnight
    service_time: int = 15  # Default service time in minutes
    
    def __post_init__(self):
        # Convert to float if strings were provided
        if isinstance(self.latitude, str):
            self.latitude = float(self.latitude)
        if isinstance(self.longitude, str):
            self.longitude = float(self.longitude)

@dataclass
class OptimizationResult:
    """Data Transfer Object representing the result of route optimization."""
    status: str
    routes: List[List[str]] = field(default_factory=list)
    total_distance: float = 0.0
    total_cost: float = 0.0 # This will be populated by RouteStatsService or solver
    assigned_vehicles: Dict[str, int] = field(default_factory=dict)
    unassigned_deliveries: List[str] = field(default_factory=list)
    detailed_routes: List[Dict[str, Any]] = field(default_factory=list) # List of dicts as per current DTO
    statistics: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> 'OptimizationResult': # Use forward reference for return type
        """
        Creates an OptimizationResult instance from a dictionary.
        Handles None input and provides default values for missing keys.
        """
        if data is None:
            logger.warning("Attempted to create OptimizationResult from None data.")
            return OptimizationResult(
                status='error',
                routes=[],
                total_distance=0.0,
                total_cost=0.0,
                assigned_vehicles={},
                unassigned_deliveries=[],
                detailed_routes=[],
                statistics={'error': 'Input data for OptimizationResult was None'}
            )
        
        try:
            # Ensure all fields have defaults if not present in the dict
            return OptimizationResult(
                status=data.get('status', 'unknown'),
                routes=data.get('routes', []),
                total_distance=data.get('total_distance', 0.0),
                total_cost=data.get('total_cost', 0.0),
                assigned_vehicles=data.get('assigned_vehicles', {}),
                unassigned_deliveries=data.get('unassigned_deliveries', []),
                detailed_routes=data.get('detailed_routes', []),
                statistics=data.get('statistics', {})
            )
        except Exception as e: # Catch any unexpected error during attribute access or .get if data is not a dict
            logger.error(f"Failed to convert dictionary to OptimizationResult: {e}", exc_info=True)
            return OptimizationResult(
                status='error',
                routes=[],
                total_distance=0.0,
                total_cost=0.0,
                assigned_vehicles={},
                unassigned_deliveries=[], # Or attempt to get from data if possible
                detailed_routes=[],
                statistics={'error': f"Conversion error from dict: {str(e)}"}
            )

@dataclass
class RouteSegment:
    """Represents a segment of a route between two locations."""
    from_location: str
    to_location: str
    path: List[str]
    distance: float
    estimated_time: Optional[float] = None

@dataclass
class DetailedRoute:
    """Represents a detailed route for a vehicle."""
    vehicle_id: str
    stops: List[str] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    capacity_utilization: float = 0.0
    estimated_arrival_times: Dict[str, int] = field(default_factory=dict)

@dataclass
class ReroutingInfo:
    """Information about a rerouting operation."""
    reason: str
    traffic_factors: int = 0 # Count of traffic factors applied
    completed_deliveries: int = 0
    remaining_deliveries: int = 0
    delay_locations: List[str] = field(default_factory=list) # Actual IDs of delayed locations
    blocked_segments: List[Tuple[str, str]] = field(default_factory=list) # Actual (from,to) tuples of blocked segments
    
def validate_optimization_result(result: Dict[str, Any]) -> bool:
    """
    Validate the optimization result structure.
    
    Args:
        result: The optimization result to validate
        
    Returns:
        True if the result is valid, False otherwise
        
    Raises:
        ValueError: If the result is invalid with a specific message
    """
    # Check required top-level fields
    required_fields = ['status']
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field: {field}")
    
    # Check status field
    if result['status'] not in ['success', 'failed']:
        raise ValueError(f"Invalid status value: {result['status']}")
    
    # If status is failed, no further validation needed
    if result['status'] == 'failed':
        return True
        
    # Check routes if status is success
    if 'routes' not in result:
        raise ValueError("Missing 'routes' in successful result")
    
    if not isinstance(result['routes'], list):
        raise ValueError("'routes' must be a list")
    
    # Validate assigned_vehicles if present
    if 'assigned_vehicles' in result:
        if not isinstance(result['assigned_vehicles'], dict):
            raise ValueError("'assigned_vehicles' must be a dictionary")
            
        # Check that route indices are valid
        for vehicle_id, route_idx in result['assigned_vehicles'].items():
            if not isinstance(route_idx, int) or route_idx < 0 or route_idx >= len(result['routes']):
                raise ValueError(f"Invalid route index {route_idx} for vehicle {vehicle_id}")
    
    # Validate detailed_routes if present
    if 'detailed_routes' in result:
        if not isinstance(result['detailed_routes'], list):
            raise ValueError("'detailed_routes' must be a list")
            
        # Check each detailed route has required fields
        for route_idx, route in enumerate(result['detailed_routes']):
            if not isinstance(route, dict):
                raise ValueError(f"Route at index {route_idx} must be a dictionary")
                
            if 'vehicle_id' not in route:
                raise ValueError(f"Missing 'vehicle_id' in route at index {route_idx}")
            
            if 'stops' not in route and 'segments' not in route:
                raise ValueError(f"Route at index {route_idx} must have either 'stops' or 'segments'")
                
            # Validate segments if present
            if 'segments' in route:
                for seg_idx, segment in enumerate(route['segments']):
                    if not isinstance(segment, dict):
                        raise ValueError(f"Segment at index {seg_idx} in route {route_idx} must be a dictionary")
                        
                    for field in ['from', 'to', 'distance']:
                        if field not in segment:
                            raise ValueError(f"Missing '{field}' in segment {seg_idx} of route {route_idx}")
    
    return True

