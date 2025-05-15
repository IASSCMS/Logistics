import logging 
from math import radians, cos, sin, asin, sqrt
from typing import Any, Dict, List, Tuple
import numpy as np

from route_optimizer.core.constants import MAX_SAFE_DISTANCE
from route_optimizer.core.distance_matrix import DistanceMatrixBuilder
from route_optimizer.core.types_1 import Location

logger = logging.getLogger(__name__)

class TrafficService:
    def __init__(self, api_key=None):
        """
        Initialize the traffic service with optional API key.
        
        Args:
            api_key: API key for external services if applicable
        """
        self.api_key = api_key
    
    @staticmethod
    def apply_traffic_factors(
        distance_matrix: np.ndarray,
        traffic_data: Dict[Tuple[int, int], float]
    ) -> np.ndarray:
        # Calls the robust, consolidated method in DistanceMatrixBuilder
        return DistanceMatrixBuilder.add_traffic_factors(distance_matrix, traffic_data)

    def _calculate_distance_haversine(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate the Haversine distance between two locations.
        Returns distance in kilometers.
        """
        # Check for attribute existence and ensure coordinates are not None
        if (hasattr(loc1, 'latitude') and loc1.latitude is not None and
            hasattr(loc1, 'longitude') and loc1.longitude is not None and
            hasattr(loc2, 'latitude') and loc2.latitude is not None and
            hasattr(loc2, 'longitude') and loc2.longitude is not None):
            
            # All coordinates are present and not None
            # Ensure they are explicitly floats before passing to haversine
            try:
                lat1_f = float(loc1.latitude)
                lon1_f = float(loc1.longitude)
                lat2_f = float(loc2.latitude)
                lon2_f = float(loc2.longitude)
                return DistanceMatrixBuilder._haversine_distance(lat1_f, lon1_f, lat2_f, lon2_f)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert coordinates to float for Haversine distance between {loc1.id} and {loc2.id}.")
                return float('inf')
        
        logger.warning(f"Could not calculate Haversine distance between {loc1.id} and {loc2.id} due to missing or invalid coordinates.")
        return float('inf')
    
    def create_road_graph(self, locations: List[Location]) -> Dict[str, Any]:
        """
        Create a road graph from a list of locations.
        If an API key is provided, it attempts to use the Google Distance Matrix API
        to get actual road distances and travel times. Otherwise, it falls back to
        Haversine distances.

        Args:
            locations: List of Location objects
            
        Returns:
            Graph representation with nodes and edges. Edges will contain
            {'distance': float, 'time': Optional[float], 'polyline': Optional[str]}.
            Distance is in km, time in seconds.
        """
        graph = {'nodes': {}, 'edges': {}}
        if not locations:
            return graph

        for location in locations:
            graph['nodes'][location.id] = location
        
        location_ids = [loc.id for loc in locations]
        num_locations = len(locations)

        if self.api_key:
            logger.info(f"Attempting to create road graph using API for {num_locations} locations.")
            try:
                # Now expects three values if create_distance_matrix_from_api is updated
                api_dist_matrix_km, api_time_matrix_sec, returned_location_ids = \
                    DistanceMatrixBuilder.create_distance_matrix_from_api(
                        locations, self.api_key, use_cache=True
                    )

                if returned_location_ids != location_ids:
                    logger.error("Location ID mismatch...")
                    raise ValueError("Location ID mismatch")

                for i in range(num_locations):
                    from_loc_id = location_ids[i]
                    graph['edges'][from_loc_id] = {}
                    for j in range(num_locations):
                        if i == j:
                            continue
                        to_loc_id = location_ids[j]
                        graph['edges'][from_loc_id][to_loc_id] = {
                            'distance': api_dist_matrix_km[i, j],
                            'time': api_time_matrix_sec[i, j] if api_time_matrix_sec is not None else None, # Time in seconds
                            'polyline': None 
                        }
                logger.info("Successfully created road graph using API-derived distances and times.")

            except Exception as e:
                logger.error(f"API call failed for create_road_graph: {e}. Falling back to Haversine distances.")
                # Fallback to Haversine if API fails
                for i in range(num_locations):
                    loc1 = locations[i]
                    graph['edges'][loc1.id] = {}
                    for j in range(num_locations):
                        if i == j:
                            continue
                        loc2 = locations[j]
                        dist = self._calculate_distance_haversine(loc1, loc2)
                        graph['edges'][loc1.id][loc2.id] = {
                            'distance': dist,
                            'time': None, # Could estimate time = dist / avg_speed if needed
                            'polyline': None
                        }
        else:
            logger.info("API key not provided. Creating road graph using Haversine distances.")
            for i in range(num_locations):
                loc1 = locations[i]
                graph['edges'][loc1.id] = {}
                for j in range(num_locations):
                    if i == j:
                        continue
                    loc2 = locations[j]
                    dist = self._calculate_distance_haversine(loc1, loc2)
                    graph['edges'][loc1.id][loc2.id] = {
                        'distance': dist,
                        'time': None,
                        'polyline': None
                    }
        return graph