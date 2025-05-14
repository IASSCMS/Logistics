"""
Distance matrix utilities for route optimization.

This module provides functions to create and manipulate distance matrices
that are used in route optimization algorithms.
"""
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import time
import json
import hashlib
import requests
from datetime import datetime, timedelta
from urllib.parse import quote

from route_optimizer.core.constants import DISTANCE_SCALING_FACTOR, MAX_SAFE_DISTANCE
from route_optimizer.core.types_1 import Location
from route_optimizer.models import DistanceMatrixCache

from route_optimizer.settings import (
    CACHE_EXPIRY_DAYS,
    GOOGLE_MAPS_API_KEY, 
    GOOGLE_MAPS_API_URL,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    RETRY_DELAY_SECONDS
)

logger = logging.getLogger(__name__)


class DistanceMatrixBuilder:
    """
    Builder class for creating distance matrices used in route optimization.
    """

    @staticmethod
    def create_distance_matrix(
        locations: List[Location],
        use_haversine: bool = True,
        distance_calculation: str = None,
        use_api: bool = False,
        api_key: str = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create a distance matrix from a list of locations.

        Args:
            locations: List of Location objects.
            use_haversine: If True, use Haversine formula for distances,
                          otherwise use Euclidean distances.
            distance_calculation: String specifying calculation method ("haversine" or "euclidean")
            use_api: Whether to use an external API for distance calculation
            api_key: API key for external service if applicable

        Returns:
            Tuple containing:
            - 2D numpy array representing distances between locations
            - List of location IDs corresponding to the matrix indices
        """
        # Handle the string-based calculation method
        if distance_calculation:
            if distance_calculation == "haversine":
                use_haversine = True
            elif distance_calculation == "euclidean":
                use_haversine = False
        
        # Handle API-based calculation if requested
        if use_api and api_key:
            try:
                # Call the API implementation 
                return DistanceMatrixBuilder.create_distance_matrix_from_api(
                    locations=locations, 
                    api_key=api_key, 
                    use_cache=True
                )
            except Exception as e:
                logger.warning(f"API distance calculation failed: {e}. Falling back to haversine.")
        
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
    def _get_api_key():
        """
        Get the Google Maps API key from settings.
        
        Returns:
            str: The API key
        """
        from route_optimizer.settings import GOOGLE_MAPS_API_KEY
        return GOOGLE_MAPS_API_KEY
                                
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
    
    @staticmethod
    def _send_request(origin_addresses, dest_addresses, api_key):
        """Builds and sends request for the given origin and destination addresses."""
        def build_address_str(addresses):
            # Build a pipe-separated string of addresses
            return '|'.join(addresses)

        import requests
        import json
        
        request = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=metric'
        origin_address_str = build_address_str(origin_addresses)
        dest_address_str = build_address_str(dest_addresses)
        request = (request + '&origins=' + origin_address_str + 
                '&destinations=' + dest_address_str + '&key=' + api_key)
        
        response = requests.get(request)
        return json.loads(response.text)

    @staticmethod
    def _process_api_response(response: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Process the Google Maps Distance Matrix API response into distance and time matrices.
        
        Args:
            response: The response from the Google Maps API
            
        Returns:
            Tuple containing (distance_matrix, time_matrix)
            Both matrices are in meters and seconds respectively
        """
        distance_matrix = []
        time_matrix = []
        
        for row in response.get('rows', []):
            dist_row = []
            time_row = []
            
            for element in row.get('elements', []):
                # Check if the element has the expected structure
                if element.get('status') == 'OK':
                    dist_row.append(element.get('distance', {}).get('value', 0))  # meters
                    time_row.append(element.get('duration', {}).get('value', 0))  # seconds
                else:
                    # For unreachable destinations, use a large value
                    logger.warning(f"Destination unreachable: {element.get('status')}")
                    dist_row.append(float('inf'))
                    time_row.append(float('inf'))
            
            distance_matrix.append(dist_row)
            time_matrix.append(time_row)
        
        return distance_matrix, time_matrix

    def _sanitize_distance_matrix(self, matrix):
        """
        Sanitize distance matrix by replacing infinite or extreme values.
        
        Args:
            matrix: Distance matrix to sanitize
            
        Returns:
            Sanitized matrix
        """
        if matrix is None:
            return np.zeros((1, 1))
        
        # Make a copy to avoid modifying the original
        sanitized = np.array(matrix, dtype=float)
        
        # Define the maximum safe distance value
        max_safe_value = MAX_SAFE_DISTANCE  # This should be defined in your constants
        
        # Replace any NaN values with a large but valid distance
        sanitized = np.nan_to_num(sanitized, nan=max_safe_value)
        
        # Replace any infinite values with a large but valid distance
        sanitized[np.isinf(sanitized)] = max_safe_value
        
        # Cap any excessively large values
        sanitized[sanitized > max_safe_value] = max_safe_value
        
        # Ensure all values are non-negative
        sanitized[sanitized < 0] = 0
        
        return sanitized

    def _apply_traffic_safely(self, distance_matrix, traffic_data):
        """
        Apply traffic factors to distance matrix with bounds checking.
        
        Args:
            distance_matrix: Original distance matrix
            traffic_data: Dictionary mapping (from_idx, to_idx) to traffic factors
            
        Returns:
            Updated distance matrix
        """
        # Make a copy to avoid modifying the original
        matrix_with_traffic = np.array(distance_matrix, dtype=float)
        
        # Get matrix dimensions
        rows, cols = matrix_with_traffic.shape
        
        # Define maximum safe factor to prevent overflow
        max_safe_factor = 5.0  # Adjust this value based on your use case
        
        for (from_idx, to_idx), factor in traffic_data.items():
            # Validate indices
            if 0 <= from_idx < rows and 0 <= to_idx < cols:
                # Validate factor (ensure it's within reasonable bounds)
                safe_factor = min(max(float(factor), 1.0), max_safe_factor)
                
                # Apply the factor
                matrix_with_traffic[from_idx, to_idx] *= safe_factor
                
                # Log if factor was capped
                if safe_factor != factor:
                    logger.warning(f"Traffic factor capped from {factor} to {safe_factor} for route ({from_idx},{to_idx})")
        
        return matrix_with_traffic

    @staticmethod
    def _build_distance_matrix(response):
        """Builds distance matrix from API response."""
        distance_matrix = []
        for row in response['rows']:
            row_list = [row['elements'][j]['distance']['value'] for j in range(len(row['elements']))]
            distance_matrix.append(row_list)
        return distance_matrix

    @staticmethod
    def create_distance_matrix_from_api(
        locations: List[Location], 
        api_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create a distance matrix from a list of locations using Google Distance Matrix API.
        Falls back to Haversine calculation if API fails.
        
        Args:
            locations: List of Location objects.
            api_key: Google API key with Distance Matrix API enabled.
            use_cache: Whether to use cached results.
            
        Returns:
            Tuple containing:
            - 2D numpy array representing distances between locations in km
            - List of location IDs corresponding to the matrix indices
        """
        # Use provided API key or fall back to settings
        api_key = api_key or GOOGLE_MAPS_API_KEY
        
        if not api_key:
            logger.warning("No Google Maps API key provided. Falling back to Haversine distance.")
            return DistanceMatrixBuilder.create_distance_matrix(locations, use_haversine=True)
        
        try:
            # Try to get from cache first if use_cache is True
            if use_cache:
                cached_result = DistanceMatrixBuilder.get_cached_matrix(locations)
                if cached_result:
                    logger.info("Using cached distance matrix")
                    return cached_result
            
            # Extract addresses from locations
            addresses = []
            location_ids = []
            for loc in locations:
                address = DistanceMatrixBuilder._format_address(loc)
                addresses.append(address)
                location_ids.append(str(loc.id))  # Convert to string for consistency
            
            # Prepare data for the API
            data = {"addresses": addresses, "API_key": api_key}
            
            # Get the matrices from Google API
            api_matrix, time_matrix = DistanceMatrixBuilder._fetch_distance_and_time_matrices(data)
            
            # Convert to numpy array (and convert from meters to kilometers)
            num_locations = len(locations)
            distance_matrix = np.zeros((num_locations, num_locations))
            
            time_matrix_np = np.array(time_matrix)
            
            # Cache the result
            if use_cache:
                DistanceMatrixBuilder.cache_matrix(distance_matrix, location_ids, time_matrix)
                
            return distance_matrix, time_matrix_np, location_ids
            
        except Exception as e:
            logger.error(f"Error creating distance matrix from API: {str(e)}")
            logger.info("Falling back to Haversine distance calculation")
            return DistanceMatrixBuilder.create_distance_matrix(locations, use_haversine=True)
    
    @staticmethod
    def _format_address(location: Location) -> str:
        """Format location address for API request."""
        components = []
        if hasattr(location, 'street_address') and location.street_address:
            components.append(location.street_address)
        if hasattr(location, 'city') and location.city:
            components.append(location.city)
        if hasattr(location, 'state') and location.state:
            components.append(location.state)
        if hasattr(location, 'zip_code') and location.zip_code:
            components.append(location.zip_code)
        
        # Join and encode address components
        address = ' '.join(components)
        return address.replace(' ', '+')
    
    @staticmethod
    def _fetch_distance_and_time_matrices(data: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Fetches distance and time matrices from Google Distance Matrix API.
        Implements retry logic with exponential backoff.
        """
        addresses = data["addresses"]
        api_key = data["API_key"]
        
        # Distance Matrix API only accepts 100 elements per request
        max_elements = 100
        num_addresses = len(addresses)
        max_rows = max_elements // num_addresses
        q, r = divmod(num_addresses, max_rows)
        dest_addresses = addresses
        distance_matrix = []
        time_matrix = []
        
        # Send q requests, returning max_rows rows per request
        for i in range(q):
            origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
            response = DistanceMatrixBuilder._send_request_with_retry(origin_addresses, dest_addresses, api_key)
            distance_rows, time_rows = DistanceMatrixBuilder._process_api_response(response)
            distance_matrix.extend(distance_rows)
            time_matrix.extend(time_rows)

        # And also in the r > 0 block
        if r > 0:
            origin_addresses = addresses[q * max_rows: q * max_rows + r]
            response = DistanceMatrixBuilder._send_request_with_retry(origin_addresses, dest_addresses, api_key)
            distance_rows, time_rows = DistanceMatrixBuilder._process_api_response(response)
            distance_matrix.extend(distance_rows)
            time_matrix.extend(time_rows)
        
        return distance_matrix, time_matrix

    @staticmethod
    def _send_request_with_retry(origin_addresses, dest_addresses, api_key):
        """Sends request with retry logic using exponential backoff."""
        retry_count = 0
        delay = RETRY_DELAY_SECONDS
        
        while retry_count < MAX_RETRIES:
            try:
                response = DistanceMatrixBuilder._send_request(origin_addresses, dest_addresses, api_key)
                
                # Check if the API returned an error
                if response.get('status') != 'OK':
                    error_message = response.get('error_message', 'Unknown API error')
                    logger.warning(f"API error: {error_message}")
                    
                    # If OVER_QUERY_LIMIT, use backoff strategy
                    if response.get('status') == 'OVER_QUERY_LIMIT':
                        retry_count += 1
                        sleep_time = delay * (BACKOFF_FACTOR ** (retry_count - 1))
                        logger.info(f"Rate limit exceeded, retrying in {sleep_time} seconds")
                        time.sleep(sleep_time)
                        continue
                    
                    # For other errors, raise exception to trigger fallback
                    raise Exception(f"Google Maps API error: {error_message}")
                
                # If we got here, the request was successful
                return response
                
            except requests.RequestException as e:
                logger.warning(f"Request failed: {str(e)}")
                retry_count += 1
                
                if retry_count < MAX_RETRIES:
                    sleep_time = delay * (BACKOFF_FACTOR ** (retry_count - 1))
                    logger.info(f"Retrying in {sleep_time} seconds")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached. Falling back to alternative method.")
                    raise
        
        # If we get here, all retries failed
        raise Exception("All API request retries failed")

    @staticmethod
    def _send_request(origin_addresses, dest_addresses, api_key):
        """Builds and sends request for the given origin and destination addresses."""
        def build_address_str(addresses):
            # Build a pipe-separated string of addresses
            return '|'.join(addresses)

        request = f"{GOOGLE_MAPS_API_URL}?units=metric"
        origin_address_str = build_address_str(origin_addresses)
        dest_address_str = build_address_str(dest_addresses)
        request = (request + '&origins=' + quote(origin_address_str) + 
                  '&destinations=' + quote(dest_address_str) + '&key=' + api_key)
        
        response = requests.get(request, timeout=10)  # 10-second timeout
        return response.json()

    @staticmethod
    def _build_distance_and_time_matrices(response):
        """Builds distance and time matrices from API response."""
        distance_matrix = []
        time_matrix = []
        
        for row in response.get('rows', []):
            dist_row = []
            time_row = []
            
            for element in row.get('elements', []):
                # Check if the element has the expected structure
                if element.get('status') == 'OK':
                    dist_row.append(element.get('distance', {}).get('value', 0))  # meters
                    time_row.append(element.get('duration', {}).get('value', 0))  # seconds
                else:
                    # For unreachable destinations, use a large value
                    logger.warning(f"Destination unreachable: {element.get('status')}")
                    dist_row.append(float('inf'))
                    time_row.append(float('inf'))
                    
            distance_matrix.append(dist_row)
            time_matrix.append(time_row)
            
        return distance_matrix, time_matrix
    
    @staticmethod
    def get_cached_matrix(locations, cache_expiry_days=None):
        """Get distance matrix from cache if available and not expired."""
        from django.db import models
        
        if not cache_expiry_days:
            cache_expiry_days = CACHE_EXPIRY_DAYS
            
        # Create a unique identifier based on the location IDs
        location_ids = sorted([str(loc.id) for loc in locations])
        cache_key = hashlib.md5(json.dumps(location_ids).encode()).hexdigest()
        
        try:
            # Check if we have this matrix cached and not expired
            cached_result = DistanceMatrixCache.objects.filter(
                cache_key=cache_key,
                created_at__gte=datetime.now() - timedelta(days=cache_expiry_days)
            ).first()
            
            if cached_result:
                distance_matrix = np.array(json.loads(cached_result.matrix_data))
                location_ids = json.loads(cached_result.location_ids)
                time_matrix = None
                if cached_result.time_matrix_data:
                    time_matrix = np.array(json.loads(cached_result.time_matrix_data))
                return distance_matrix, time_matrix, location_ids
        except (models.ObjectDoesNotExist, Exception) as e:
            logger.warning(f"Error retrieving from cache: {str(e)}")
            
        return None
    
    @staticmethod
    def cache_matrix(distance_matrix, location_ids, time_matrix=None):
        """Cache the distance and time matrices."""
        try:
            # Create a unique identifier based on the location IDs
            cache_key = hashlib.md5(json.dumps(sorted(location_ids)).encode()).hexdigest()
            
            # Convert numpy array to list for JSON serialization
            if isinstance(distance_matrix, np.ndarray):
                distance_matrix_list = distance_matrix.tolist()
            else:
                distance_matrix_list = distance_matrix
                
            # Create or update cache entry
            DistanceMatrixCache.objects.update_or_create(
                cache_key=cache_key,
                defaults={
                    'matrix_data': json.dumps(distance_matrix_list),
                    'location_ids': json.dumps(location_ids),
                    'time_matrix_data': json.dumps(time_matrix) if time_matrix else None,
                    'created_at': datetime.now()
                }
            )
        except Exception as e:
            logger.warning(f"Error caching matrix: {str(e)}")

    @staticmethod
    def _fetch_distance_matrix(data):
        """Fetches distance matrix from Google Distance Matrix API."""
        addresses = data["addresses"]
        api_key = data["API_key"]
        
        # Distance Matrix API only accepts 100 elements per request
        max_elements = 100
        num_addresses = len(addresses)
        max_rows = max_elements // num_addresses
        q, r = divmod(num_addresses, max_rows)
        dest_addresses = addresses
        distance_matrix = []
        
        # Send q requests, returning max_rows rows per request
        for i in range(q):
            origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
            response = DistanceMatrixBuilder._send_request(origin_addresses, dest_addresses, api_key)
            distance_matrix += DistanceMatrixBuilder._build_distance_matrix(response)

        # Get the remaining r rows, if necessary
        if r > 0:
            origin_addresses = addresses[q * max_rows: q * max_rows + r]
            response = DistanceMatrixBuilder._send_request(origin_addresses, dest_addresses, api_key)
            distance_matrix += DistanceMatrixBuilder._build_distance_matrix(response)
        
        return distance_matrix
