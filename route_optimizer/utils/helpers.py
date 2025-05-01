"""
Helper functions for the route optimizer module.

This module provides various utility functions used across the route optimizer.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import numpy as np
import datetime
import json
from math import radians, cos, sin, asin, sqrt

# Set up logging
logger = logging.getLogger(__name__)


def convert_minutes_to_time_str(minutes_from_midnight: int) -> str:
    """
    Convert minutes from midnight to a time string (HH:MM).
    
    Args:
        minutes_from_midnight: Minutes from midnight.
        
    Returns:
        Time string in HH:MM format.
    """
    hours, minutes = divmod(minutes_from_midnight, 60)
    return f"{hours:02d}:{minutes:02d}"


def convert_time_str_to_minutes(time_str: str) -> int:
    """
    Convert a time string (HH:MM) to minutes from midnight.
    
    Args:
        time_str: Time string in HH:MM format.
        
    Returns:
        Minutes from midnight.
    """
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except (ValueError, AttributeError):
        logger.error(f"Invalid time string format: {time_str}")
        return 0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def format_route_for_display(route: List[str], location_names: Dict[str, str]) -> str:
    """
    Format a route for display, converting location IDs to names.
    
    Args:
        route: List of location IDs in the route.
        location_names: Dictionary mapping location IDs to names.
        
    Returns:
        Formatted route string.
    """
    route_with_names = [f"{location_names.get(loc_id, loc_id)}" for loc_id in route]
    return " → ".join(route_with_names)


def calculate_route_statistics(
    routes: List[Dict[str, Any]],
    vehicles: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate statistics for the routes.
    
    Args:
        routes: List of route dictionaries.
        vehicles: Dictionary of vehicles with capacities.
        
    Returns:
        Dictionary of route statistics.
    """
    statistics = {
        "total_distance": 0.0,
        "total_cost": 0.0,
        "total_time": 0.0,
        "vehicle_utilization": 0.0,
        "average_capacity_utilization": 0.0,
        "num_vehicles_used": 0,
    }
    
    capacity_utils = []
    
    for route in routes:
        statistics["total_distance"] += route.get("total_distance", 0.0)
        statistics["total_cost"] += route.get("total_cost", 0.0)
        statistics["total_time"] += route.get("total_time", 0.0)
        
        if route.get("capacity_utilization") is not None:
            capacity_utils.append(route["capacity_utilization"])
    
    statistics["num_vehicles_used"] = len(routes)
    
    if statistics["num_vehicles_used"] > 0:
        statistics["vehicle_utilization"] = statistics["num_vehicles_used"] / len(vehicles)
    
    if capacity_utils:
        statistics["average_capacity_utilization"] = sum(capacity_utils) / len(capacity_utils)
    
    return statistics


def create_distance_time_matrices(
    locations: List[Any],
    speed_km_per_hour: float = 50.0,
    use_haversine: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create distance and time matrices from a list of locations.
    
    Args:
        locations: List of Location objects.
        speed_km_per_hour: Average speed in km/h.
        use_haversine: If True, use haversine formula for calculating distances.
        
    Returns:
        Tuple containing:
        - 2D numpy array representing distances between locations in km
        - 2D numpy array representing times between locations in minutes
        - List of location IDs corresponding to the matrix indices
    """
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    time_matrix = np.zeros((num_locations, num_locations))
    location_ids = [loc.id for loc in locations]
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                if use_haversine:
                    distance = haversine_distance(
                        locations[i].latitude, locations[i].longitude,
                        locations[j].latitude, locations[j].longitude
                    )
                else:
                    # Euclidean distance as a fallback
                    distance = sqrt(
                        (locations[i].latitude - locations[j].latitude)**2 +
                        (locations[i].longitude - locations[j].longitude)**2
                    )
                
                distance_matrix[i, j] = distance
                
                # Calculate time in minutes
                time_matrix[i, j] = (distance / speed_km_per_hour) * 60
    
    return distance_matrix, time_matrix, location_ids


def apply_external_factors(
    distance_matrix: np.ndarray,
    time_matrix: np.ndarray,
    external_factors: Dict[Tuple[int, int], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply external factors like traffic or weather to distance and time matrices.
    
    Args:
        distance_matrix: Original distance matrix.
        time_matrix: Original time matrix.
        external_factors: Dictionary mapping (i,j) tuples to factors.
                         A factor of 1.0 means no change, >1.0 means slower.
        
    Returns:
        Tuple containing updated distance and time matrices.
    """
    # Create copies to avoid modifying the originals
    updated_distance_matrix = distance_matrix.copy()
    updated_time_matrix = time_matrix.copy()
    
    for (i, j), factor in external_factors.items():
        # Distance doesn't change with traffic/weather, only time does
        updated_time_matrix[i, j] *= factor
    
    return updated_distance_matrix, updated_time_matrix


def detect_isolated_nodes(graph: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Detect nodes in the graph that are isolated (have no connections).
    
    Args:
        graph: Dictionary representing the graph with format:
              {node1: {node2: distance, ...}, ...}
        
    Returns:
        List of isolated node IDs.
    """
    isolated_nodes = []
    
    for node, connections in graph.items():
        if not connections:  # No outgoing connections
            # Check if there are any incoming connections
            has_incoming = any(node in neighbors for neighbors in graph.values())
            if not has_incoming:
                isolated_nodes.append(node)
    
    return isolated_nodes


def safe_json_dumps(obj: Any) -> str:
    """
    Safely convert an object to a JSON string, handling non-serializable types.
    
    Args:
        obj: Object to convert to JSON.
        
    Returns:
        JSON string representation of the object.
    """
    def handle_non_serializable(o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if hasattr(o, '__dict__'):
            return o.__dict__
        return str(o)
    
    return json.dumps(obj, default=handle_non_serializable)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Human-readable duration string.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0 or not parts:  # Always show minutes if there are no hours
        parts.append(f"{int(minutes)}m")
    if not parts or seconds > 0:  # Show seconds if no larger units or non-zero
        parts.append(f"{int(seconds)}s")
    
    return " ".join(parts)