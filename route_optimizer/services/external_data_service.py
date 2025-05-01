"""
Service for handling external data like traffic, weather, and roadblocks.

This module provides functionality to fetch and process external data
that affects route optimization.
"""
import logging
import datetime
import random
from typing import Dict, List, Tuple, Optional, Any, Set
import json
import requests
from urllib.parse import urlencode

from route_optimizer.core.distance_matrix import Location

# Set up logging
logger = logging.getLogger(__name__)


class ExternalDataService:
    """
    Service for fetching and processing external data like traffic, weather, and roadblocks.
    """
    
    def __init__(
        self,
        traffic_api_key: Optional[str] = None,
        weather_api_key: Optional[str] = None,
        use_mocks: bool = False
    ):
        """
        Initialize the external data service.
        
        Args:
            traffic_api_key: API key for traffic data service.
            weather_api_key: API key for weather data service.
            use_mocks: Whether to use mock data instead of real API calls.
        """
        self.traffic_api_key = traffic_api_key
        self.weather_api_key = weather_api_key
        self.use_mocks = use_mocks
    
    def get_traffic_data(
        self,
        locations: List[Location]
    ) -> Dict[Tuple[int, int], float]:
        """
        Get current traffic data for the routes between locations.
        
        Args:
            locations: List of Location objects.
        
        Returns:
            Dictionary mapping (from_idx, to_idx) tuples to traffic factors.
            A factor of 1.0 means normal traffic, >1.0 means slower.
        """
        if self.use_mocks:
            return self._mock_traffic_data(locations)
        
        try:
            # In a real implementation, this would call an external traffic API
            # For now, we'll just return mock data
            logger.warning("Real traffic API not implemented, using mock data")
            return self._mock_traffic_data(locations)
        except Exception as e:
            logger.error(f"Error fetching traffic data: {str(e)}")
            # Return empty data in case of error
            return {}
    
    def get_weather_data(
        self,
        locations: List[Location]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current weather data for the given locations.
        
        Args:
            locations: List of Location objects.
        
        Returns:
            Dictionary mapping location IDs to weather data.
        """
        if self.use_mocks:
            return self._mock_weather_data(locations)
        
        try:
            # In a real implementation, this would call an external weather API
            # For now, we'll just return mock data
            logger.warning("Real weather API not implemented, using mock data")
            return self._mock_weather_data(locations)
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            # Return empty data in case of error
            return {}
    
    def get_roadblock_data(
        self,
        locations: List[Location]
    ) -> List[Tuple[str, str]]:
        """
        Get current roadblock data for the routes between locations.
        
        Args:
            locations: List of Location objects.
        
        Returns:
            List of tuples (location_id1, location_id2) representing blocked roads.
        """
        if self.use_mocks:
            return self._mock_roadblock_data(locations)
        
        try:
            # In a real implementation, this would call an external roadblock API
            # For now, we'll just return mock data
            logger.warning("Real roadblock API not implemented, using mock data")
            return self._mock_roadblock_data(locations)
        except Exception as e:
            logger.error(f"Error fetching roadblock data: {str(e)}")
            # Return empty data in case of error
            return []
    
    def _mock_traffic_data(
        self,
        locations: List[Location]
    ) -> Dict[Tuple[int, int], float]:
        """
        Generate mock traffic data for testing.
        
        Args:
            locations: List of Location objects.
        
        Returns:
            Dictionary mapping (from_idx, to_idx) tuples to traffic factors.
        """
        traffic_data = {}
        num_locations = len(locations)
        
        # Generate random traffic factors for about 30% of the routes
        num_traffic_entries = int(0.3 * num_locations * (num_locations - 1))
        
        for _ in range(num_traffic_entries):
            from_idx = random.randint(0, num_locations - 1)
            to_idx = random.randint(0, num_locations - 1)
            
            # Ensure we don't have self-loops
            if from_idx != to_idx:
                # Traffic factor between 1.0 (normal) and 2.0 (heavy traffic)
                traffic_factor = 1.0 + random.random()
                traffic_data[(from_idx, to_idx)] = traffic_factor
        
        return traffic_data
    
    def _mock_weather_data(
        self,
        locations: List[Location]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate mock weather data for testing.
        
        Args:
            locations: List of Location objects.
        
        Returns:
            Dictionary mapping location IDs to weather data.
        """
        weather_conditions = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Thunderstorm']
        weather_data = {}
        
        for location in locations:
            condition = random.choice(weather_conditions)
            temperature = random.uniform(-5, 35)  # Temperature in Celsius
            
            # Determine weather impacts on travel
            impact_factor = 1.0
            if condition == 'Rain':
                impact_factor = 1.2
            elif condition == 'Snow':
                impact_factor = 1.5
            elif condition == 'Thunderstorm':
                impact_factor = 1.8
            
            weather_data[location.id] = {
                'condition': condition,
                'temperature': round(temperature, 1),
                'impact_factor': impact_factor
            }
        
        return weather_data
    
    def _mock_roadblock_data(
        self,
        locations: List[Location]
    ) -> List[Tuple[str, str]]:
        """
        Generate mock roadblock data for testing.
        
        Args:
            locations: List of Location objects.
        
        Returns:
            List of tuples (location_id1, location_id2) representing blocked roads.
        """
        roadblocks = []
        num_locations = len(locations)
        
        # Generate random roadblocks for about 5% of the routes
        num_roadblocks = int(0.05 * num_locations * (num_locations - 1))
        num_roadblocks = min(num_roadblocks, 3)  # Limit the number of roadblocks
        
        for _ in range(num_roadblocks):
            idx1 = random.randint(0, num_locations - 1)
            idx2 = random.randint(0, num_locations - 1)
            
            # Ensure we don't have self-loops
            if idx1 != idx2:
                location1 = locations[idx1]
                location2 = locations[idx2]
                roadblocks.append((location1.id, location2.id))
        
        return roadblocks
    
    def calculate_weather_impact(
        self,
        weather_data: Dict[str, Dict[str, Any]],
        locations: List[Location]
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate the impact of weather on travel times.
        
        Args:
            weather_data: Weather data dictionary.
            locations: List of Location objects.
        
        Returns:
            Dictionary mapping (from_idx, to_idx) tuples to weather impact factors.
        """
        weather_impact = {}
        num_locations = len(locations)
        
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    continue  # Skip self-loops
                
                from_loc = locations[i]
                to_loc = locations[j]
                
                # Get weather impact factors for both locations
                from_factor = weather_data.get(from_loc.id, {}).get('impact_factor', 1.0)
                to_factor = weather_data.get(to_loc.id, {}).get('impact_factor', 1.0)
                
                # Take the worse of the two weather conditions
                impact_factor = max(from_factor, to_factor)
                
                # Only add if there's actually some impact
                if impact_factor > 1.0:
                    weather_impact[(i, j)] = impact_factor
        
        return weather_impact
    
    def combine_traffic_and_weather(
        self,
        traffic_data: Dict[Tuple[int, int], float],
        weather_impact: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], float]:
        """
        Combine traffic and weather impact data.
        
        Args:
            traffic_data: Traffic factor dictionary.
            weather_impact: Weather impact factor dictionary.
        
        Returns:
            Combined impact factors.
        """
        combined_data = traffic_data.copy()
        
        # Add weather impacts
        for (i, j), factor in weather_impact.items():
            if (i, j) in combined_data:
                # Multiply the impacts
                combined_data[(i, j)] *= factor
            else:
                combined_data[(i, j)] = factor
        
        return combined_data