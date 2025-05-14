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
import time # For retry delays
from requests.exceptions import RequestException, HTTPError # For specific request errors

from route_optimizer.core.types_1 import Location
from route_optimizer.settings import MAX_RETRIES, BACKOFF_FACTOR, RETRY_DELAY_SECONDS # For retry logic

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
        # Hypothetical base URLs for external APIs
        self.traffic_api_url = "https_traffic_api_example_com_v1_data" # Replace with actual URL
        self.weather_api_url = "https_weather_api_example_com_v1_current" # Replace with actual URL
        self.roadblock_api_url = "https_roadblock_api_example_com_v1_alerts" # Replace with actual URL

    def _make_api_request(self, url: str, params: Dict[str, Any], api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Helper function to make an API request with retry logic.
        """
        headers = {}
        if api_key:
            # Assuming API key is passed in a header, adjust as per actual API spec
            headers['Authorization'] = f'Bearer {api_key}'
            # Alternatively, some APIs take the key as a query parameter:
            # params['apiKey'] = api_key

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10) # 10-second timeout
                response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
                return response.json()  # Assuming API returns JSON
            except HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err} - Status: {http_err.response.status_code}")
                if http_err.response.status_code == 429: # Rate limit
                    if attempt < MAX_RETRIES - 1:
                        sleep_time = RETRY_DELAY_SECONDS * (BACKOFF_FACTOR ** attempt)
                        logger.info(f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.error("Max retries reached for rate limit.")
                        return None # Or re-raise
                elif http_err.response.status_code in [401, 403]: # Auth errors
                    logger.error("Authentication/Authorization error. Check API key.")
                    return None # No point in retrying auth errors usually
                else: # Other HTTP errors
                    # For other client/server errors, might not be worth retrying immediately
                    return None
            except RequestException as req_err: # Catches ConnectionError, Timeout, etc.
                logger.error(f"Request exception occurred: {req_err}")
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_DELAY_SECONDS * (BACKOFF_FACTOR ** attempt)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached for request exception: {req_err}")
                    return None
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON response: {json_err}")
                return None # Malformed JSON
            except Exception as e: # Catch-all for other unexpected errors
                logger.error(f"An unexpected error occurred during API request: {e}", exc_info=True)
                return None # Or re-raise if it's critical

        logger.error(f"Failed to fetch data from {url} after {MAX_RETRIES} attempts.")
        return None

    def get_traffic_data(
        self,
        locations: List[Location]
    ) -> Dict[Tuple[int, int], float]:
        """
        Get current traffic data for the routes between locations.
        """
        if self.use_mocks:
            logger.info("Using mock traffic data.")
            return self._mock_traffic_data(locations)

        if not self.traffic_api_key:
            logger.warning("Traffic API key not provided. Falling back to mock data.")
            return self._mock_traffic_data(locations)

        # Example: API expects a list of location coordinates (lat,lon)
        # This is highly dependent on the actual API design.
        # For simplicity, let's assume the API can take all locations and figure out pairs,
        # or you might need to make calls for each pair.
        
        # Hypothetical: API takes a list of 'points' as 'lat,lon|lat,lon|...'
        # points_param = "|".join([f"{loc.latitude},{loc.longitude}" for loc in locations])
        # params = {"locations": points_param, "key": self.traffic_api_key } # Or key in header

        # More realistically, for traffic between N locations, you might query for an NxN matrix
        # or specific segments. For now, let's make a generic call and assume the API returns
        # factors for pairs identified by their indices in the input.
        
        location_coords = [{"id": loc.id, "lat": loc.latitude, "lon": loc.longitude} for loc in locations]
        params = {
            # "apiKey": self.traffic_api_key # If API key is a query param
        }
        # The body might contain location_coords for a POST request, or they might be query params
        # For a GET request, complex data is often less common.

        # For this example, let's assume a simplified API:
        # It takes a general query and returns traffic factors for predefined segments
        # This is a very simplified placeholder for actual API interaction.
        # A real traffic API would likely be more complex, similar to Google Distance Matrix.

        logger.info(f"Fetching real traffic data for {len(locations)} locations...")
        # Constructing parameters for a hypothetical API that takes a list of location IDs
        # and returns traffic factors between them.
        # This is a placeholder for actual API parameter construction.
        loc_ids_param = ",".join([loc.id for loc in locations])
        request_params = {'location_ids': loc_ids_param} # Example parameter

        api_response = self._make_api_request(self.traffic_api_url, request_params, self.traffic_api_key)

        if api_response and api_response.get('status') == 'success':
            processed_traffic_data = {}
            # Hypothetical response:
            # { "status": "success", "traffic_factors": [ {"from_idx": 0, "to_idx": 1, "factor": 1.5}, ... ] }
            raw_factors = api_response.get('traffic_factors', [])
            for factor_info in raw_factors:
                from_idx = factor_info.get('from_idx')
                to_idx = factor_info.get('to_idx')
                factor = factor_info.get('factor')
                if from_idx is not None and to_idx is not None and factor is not None:
                    if 0 <= from_idx < len(locations) and 0 <= to_idx < len(locations) and from_idx != to_idx:
                        processed_traffic_data[(from_idx, to_idx)] = float(factor)
            logger.info(f"Successfully processed {len(processed_traffic_data)} traffic factors from API.")
            return processed_traffic_data
        else:
            logger.warning("Failed to fetch or process real traffic data, or API status was not 'success'. Falling back to mock data.")
            return self._mock_traffic_data(locations)

    def get_weather_data(
        self,
        locations: List[Location]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current weather data for the given locations.
        """
        if self.use_mocks:
            logger.info("Using mock weather data.")
            return self._mock_weather_data(locations)

        if not self.weather_api_key:
            logger.warning("Weather API key not provided. Falling back to mock data.")
            return self._mock_weather_data(locations)

        processed_weather_data = {}
        logger.info(f"Fetching real weather data for {len(locations)} locations...")

        for loc in locations:
            # Hypothetical: API takes latitude and longitude for each location
            params = {
                "lat": loc.latitude,
                "lon": loc.longitude,
                # "apiKey": self.weather_api_key # If API key is a query param
                "units": "metric" # Example
            }
            
            api_response = self._make_api_request(self.weather_api_url, params, self.weather_api_key)

            if api_response and api_response.get('status') == 'success':
                # Hypothetical response:
                # { "status": "success", "location_id": "loc1",
                #   "weather": { "condition": "Rain", "temperature_celsius": 10, "impact_factor": 1.2 } }
                weather_info = api_response.get('weather')
                if weather_info:
                    processed_weather_data[loc.id] = {
                        'condition': weather_info.get('condition', 'Unknown'),
                        'temperature': weather_info.get('temperature_celsius', 0),
                        'impact_factor': weather_info.get('impact_factor', 1.0)
                    }
            else:
                logger.warning(f"Failed to fetch or process weather data for location {loc.id}. Using default/mock values for this location.")
                # Fallback for a single failed location - could use mock or default
                mock_loc_weather = self._mock_weather_data([loc])
                if loc.id in mock_loc_weather:
                     processed_weather_data[loc.id] = mock_loc_weather[loc.id]
                else: # Basic default
                     processed_weather_data[loc.id] = {'condition': 'Unknown', 'temperature': 0, 'impact_factor': 1.0}


        if processed_weather_data:
             logger.info(f"Successfully processed weather data for {len(processed_weather_data)} locations from API.")
        else:
            logger.warning("Failed to fetch any real weather data. Falling back to full mock data.")
            return self._mock_weather_data(locations) # Fallback to all mock if all fail

        return processed_weather_data

    def get_roadblock_data(
        self,
        locations: List[Location] # Locations might be used to define a bounding box for query
    ) -> List[Tuple[str, str]]:
        """
        Get current roadblock data.
        """
        if self.use_mocks:
            logger.info("Using mock roadblock data.")
            return self._mock_roadblock_data(locations)
        
        # Roadblock APIs might not need an API key or might take it differently
        # For this example, we'll assume no specific key for roadblocks, or it's handled by _make_api_request if passed
        logger.info("Fetching real roadblock data...")

        # Hypothetical: API takes a geographical area or a list of route corridors
        # For simplicity, let's assume it's a general query for a region defined by your locations.
        # We'll make one call and expect a list of blocked segments by location IDs.
        
        # Example: Define a bounding box from locations
        min_lat = min(loc.latitude for loc in locations)
        max_lat = max(loc.latitude for loc in locations)
        min_lon = min(loc.longitude for loc in locations)
        max_lon = max(loc.longitude for loc in locations)
        
        params = {
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}", # southwest_lng, southwest_lat, northeast_lng, northeast_lat
            # Potentially add API key if required
        }

        api_response = self._make_api_request(self.roadblock_api_url, params, None) # Assuming no key or handled by URL

        if api_response and api_response.get('status') == 'success':
            processed_roadblocks = []
            # Hypothetical response:
            # { "status": "success", "roadblocks": [ {"from_location_id": "A", "to_location_id": "B"}, ... ] }
            raw_blocks = api_response.get('roadblocks', [])
            for block in raw_blocks:
                from_loc = block.get('from_location_id')
                to_loc = block.get('to_location_id')
                if from_loc and to_loc:
                    processed_roadblocks.append((str(from_loc), str(to_loc)))
            logger.info(f"Successfully processed {len(processed_roadblocks)} roadblocks from API.")
            return processed_roadblocks
        else:
            logger.warning("Failed to fetch or process real roadblock data. Falling back to mock data.")
            return self._mock_roadblock_data(locations)

    # --- Mock Data Generation Methods (Unchanged) ---
    def _mock_traffic_data(
        self,
        locations: List[Location]
    ) -> Dict[Tuple[int, int], float]:
        traffic_data = {}
        num_locations = len(locations)
        if num_locations <= 1: return {} # Avoid issues with range if only 0 or 1 location
        num_traffic_entries = int(0.3 * num_locations * (num_locations - 1))
        for _ in range(num_traffic_entries):
            from_idx = random.randint(0, num_locations - 1)
            to_idx = random.randint(0, num_locations - 1)
            if from_idx != to_idx:
                traffic_factor = 1.0 + random.random()
                traffic_data[(from_idx, to_idx)] = traffic_factor
        return traffic_data

    def _mock_weather_data(
        self,
        locations: List[Location]
    ) -> Dict[str, Dict[str, Any]]:
        weather_conditions = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Thunderstorm']
        weather_data = {}
        for location in locations:
            condition = random.choice(weather_conditions)
            temperature = random.uniform(-5, 35)
            impact_factor = 1.0
            if condition == 'Rain': impact_factor = 1.2
            elif condition == 'Snow': impact_factor = 1.5
            elif condition == 'Thunderstorm': impact_factor = 1.8
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
        roadblocks = []
        num_locations = len(locations)
        if num_locations <= 1: return [] # Avoid issues if only 0 or 1 location
        num_roadblocks = int(0.05 * num_locations * (num_locations - 1))
        num_roadblocks = min(num_roadblocks, 3)
        for _ in range(num_roadblocks):
            idx1 = random.randint(0, num_locations - 1)
            idx2 = random.randint(0, num_locations - 1)
            if idx1 != idx2:
                roadblocks.append((locations[idx1].id, locations[idx2].id))
        return roadblocks

    # --- Data Processing Methods (Unchanged) ---
    def calculate_weather_impact(
        self,
        weather_data: Dict[str, Dict[str, Any]],
        locations: List[Location]
    ) -> Dict[Tuple[int, int], float]:
        weather_impact = {}
        num_locations = len(locations)
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j: continue
                from_loc_id = locations[i].id
                to_loc_id = locations[j].id
                from_factor = weather_data.get(from_loc_id, {}).get('impact_factor', 1.0)
                to_factor = weather_data.get(to_loc_id, {}).get('impact_factor', 1.0)
                impact_factor = max(from_factor, to_factor)
                if impact_factor > 1.0:
                    weather_impact[(i, j)] = impact_factor
        return weather_impact

    def combine_traffic_and_weather(
        self,
        traffic_data: Dict[Tuple[int, int], float],
        weather_impact: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], float]:
        combined_data = traffic_data.copy()
        for (i, j), factor in weather_impact.items():
            if (i, j) in combined_data:
                combined_data[(i, j)] *= factor
            else:
                combined_data[(i, j)] = factor
        return combined_data