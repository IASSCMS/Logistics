"""
Distance matrix utilities for route optimization.

This module provides functions to create and manipulate distance matrices
that are used in route optimization algorithms.
"""
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Location:
    """Class representing a location with coordinates and metadata."""
    id: str
    name: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    is_depot: bool = False
    time_window_start: Optional[int] = None  # In minutes from midnight
    time_window_end: Optional[int] = None    # In minutes from midnight
    service_time: int = 15  # Default service time in minutes


class DistanceMatrixBuilder:
    """
    Builder class for creating distance matrices used in route optimization.
    """

    @staticmethod
    def create_distance_matrix(
        locations: List[Location],
        use_haversine: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create a distance matrix from a list of locations.

        Args:
            locations: List of Location objects.
            use_haversine: If True, use Haversine formula for distances,
                          otherwise use Euclidean distances.

        Returns:
            Tuple containing:
            - 2D numpy array representing distances between locations
            - List of location IDs corresponding to the matrix indices
        """
        num_locations = len(locations)
        distance_matrix = np.zeros((num_locations, num_locations))
        location_ids = [loc.id for loc in locations]
        
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    continue  # Zero distance to self
                
                if use_haversine:
                    distance = DistanceMatrixBuilder._haversine_distance(
                        locations[i].latitude, locations[i].longitude,
                        locations[j].latitude, locations[j].longitude
                    )
                else:
                    distance = DistanceMatrixBuilder._euclidean_distance(
                        locations[i].latitude, locations[i].longitude,
                        locations[j].latitude, locations[j].longitude
                    )
                
                distance_matrix[i, j] = distance
        
        return distance_matrix, location_ids

    @staticmethod
    def distance_matrix_to_graph(
        distance_matrix: np.ndarray,
        location_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert a distance matrix to a graph representation for Dijkstra's algorithm.

        Args:
            distance_matrix: 2D numpy array of distances.
            location_ids: List of location IDs corresponding to matrix indices.

        Returns:
            Dictionary representing the graph with format:
            {node1: {node2: distance, ...}, ...}
        """
        graph = {}
        
        for i, from_id in enumerate(location_ids):
            if from_id not in graph:
                graph[from_id] = {}
            
            for j, to_id in enumerate(location_ids):
                if i != j:  # Skip self-connections
                    graph[from_id][to_id] = distance_matrix[i, j]
        
        return graph

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r

    @staticmethod
    def _euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Euclidean distance between two points.
        This is useful for testing and as a fallback.

        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point

        Returns:
            Euclidean distance between the points
        """
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

    @staticmethod
    def add_traffic_factors(
        distance_matrix: np.ndarray,
        traffic_factors: Dict[Tuple[int, int], float]
    ) -> np.ndarray:
        """
        Apply traffic factors to a distance matrix.

        Args:
            distance_matrix: Original distance matrix
            traffic_factors: Dictionary mapping (i,j) tuples to traffic factors.
                            A factor of 1.0 means no change, >1.0 means slower.

        Returns:
            Updated distance matrix with traffic factors applied
        """
        matrix_with_traffic = distance_matrix.copy()
        
        for (i, j), factor in traffic_factors.items():
            matrix_with_traffic[i, j] *= factor
            
        return matrix_with_traffic