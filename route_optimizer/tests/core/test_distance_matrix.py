import unittest
from unittest.mock import patch, MagicMock
from urllib.parse import unquote
import numpy as np
import requests
import json
from datetime import datetime, timedelta

from route_optimizer.core.distance_matrix import DistanceMatrixBuilder, Location
from route_optimizer.core.constants import DISTANCE_SCALING_FACTOR, MAX_SAFE_DISTANCE, MAX_SAFE_TIME

class TestDistanceMatrixBuilder(unittest.TestCase):
    """Test cases for DistanceMatrixBuilder."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = DistanceMatrixBuilder()
        
        # Sample locations
        self.locations = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True),
            Location(id="customer1", name="Customer 1", latitude=1.0, longitude=1.0),
            Location(id="customer2", name="Customer 2", latitude=2.0, longitude=2.0),
            Location(id="customer3", name="Customer 3", latitude=3.0, longitude=3.0)
        ]

    def test_haversine_distance(self):
        """Test the Haversine distance calculation."""
        # Test distance from (0,0) to (1,1)
        dist = self.builder._haversine_distance(0.0, 0.0, 1.0, 1.0)
        # Approximate distance in km between these coordinates is ~157 km
        self.assertAlmostEqual(dist, 157.2, delta=1.0)
        
        # Test zero distance
        dist = self.builder._haversine_distance(1.0, 1.0, 1.0, 1.0)
        self.assertEqual(dist, 0.0)

    def test_euclidean_distance(self):
        """Test the Euclidean distance calculation."""
        # Test distance from (0,0) to (3,4)
        dist = self.builder._euclidean_distance(0.0, 0.0, 3.0, 4.0)
        self.assertEqual(dist, 5.0)
        
        # Test zero distance
        dist = self.builder._euclidean_distance(1.0, 1.0, 1.0, 1.0)
        self.assertEqual(dist, 0.0)

    def test_create_distance_matrix_euclidean(self):
        """Test creating a distance matrix using Euclidean distance."""
        # Test without average_speed_kmh (time_matrix should be None)
        dist_matrix, time_matrix, location_ids = self.builder.create_distance_matrix(
            self.locations,
            distance_calculation="euclidean",
            average_speed_kmh=None # Explicitly None
        )

        self.assertEqual(dist_matrix.shape, (4, 4))
        self.assertIsNone(time_matrix) # Expect None if no speed is provided
        self.assertEqual(location_ids, ["depot", "customer1", "customer2", "customer3"])
        self.assertAlmostEqual(dist_matrix[0, 1], 1.414, delta=0.001) # (0,0) to (1,1)
        for i in range(4):
            self.assertEqual(dist_matrix[i, i], 0.0)

        # Test with average_speed_kmh (time_matrix should be estimated)
        dist_matrix_2, time_matrix_2, location_ids_2 = self.builder.create_distance_matrix(
            self.locations,
            distance_calculation="euclidean",
            average_speed_kmh=60 # e.g., 60 km/h
        )
        self.assertEqual(dist_matrix_2.shape, (4, 4))
        self.assertIsNotNone(time_matrix_2)
        self.assertEqual(time_matrix_2.shape, (4, 4))
        self.assertEqual(location_ids_2, ["depot", "customer1", "customer2", "customer3"])
        # Time for (0,0) to (1,1) = 1.414 km / 60 km/h * 60 min/h = 1.414 minutes
        self.assertAlmostEqual(time_matrix_2[0, 1], 1.414, delta=0.001)
        for i in range(4):
            self.assertEqual(time_matrix_2[i, i], 0.0)

    def test_create_distance_matrix_haversine(self):
        """Test creating a distance matrix using Haversine distance."""
        # Test without average_speed_kmh
        dist_matrix, time_matrix, location_ids = self.builder.create_distance_matrix(
            self.locations,
            distance_calculation="haversine", # or use_haversine=True
            average_speed_kmh=None
        )

        self.assertEqual(dist_matrix.shape, (4, 4))
        self.assertIsNone(time_matrix)
        self.assertEqual(location_ids, ["depot", "customer1", "customer2", "customer3"])
        # Depot (0,0) to Customer1 (1,1) is ~157.2 km
        self.assertAlmostEqual(dist_matrix[0, 1], 157.2, delta=1.0)
        for i in range(4):
            self.assertEqual(dist_matrix[i, i], 0.0)

        # Test with average_speed_kmh
        dist_matrix_2, time_matrix_2, location_ids_2 = self.builder.create_distance_matrix(
            self.locations,
            distance_calculation="haversine",
            average_speed_kmh=100 # e.g., 100 km/h
        )
        self.assertEqual(dist_matrix_2.shape, (4, 4))
        self.assertIsNotNone(time_matrix_2)
        self.assertEqual(time_matrix_2.shape, (4, 4))
        self.assertEqual(location_ids_2, ["depot", "customer1", "customer2", "customer3"])
        # Time for (0,0) to (1,1) = 157.2 km / 100 km/h * 60 min/h = 94.32 minutes
        self.assertAlmostEqual(time_matrix_2[0, 1], (157.2 / 100.0) * 60.0, delta=1.0)
        for i in range(4):
            self.assertEqual(time_matrix_2[i, i], 0.0)

    def test_process_api_response(self):
        """Test processing of Google API response data, ensuring km and minutes."""
        mock_response = {
            'rows': [
                {'elements': [
                    {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}}, # 10km, 10min
                    {'status': 'OK', 'distance': {'value': 20000}, 'duration': {'value': 1200}} # 20km, 20min
                ]},
                {'elements': [
                    {'status': 'OK', 'distance': {'value': 30000}, 'duration': {'value': 1800}}, # 30km, 30min
                    {'status': 'OK', 'distance': {'value': 5000}, 'duration': {'value': 300}}   # 5km, 5min
                ]}
            ]
        }
        distance_matrix_list_km, time_matrix_list_min = DistanceMatrixBuilder._process_api_response(mock_response)

        expected_distances_km = [[10.0, 20.0], [30.0, 5.0]]
        expected_times_min = [[10.0, 20.0], [30.0, 5.0]]

        self.assertEqual(distance_matrix_list_km, expected_distances_km)
        self.assertEqual(time_matrix_list_min, expected_times_min)

    def test_process_api_response_with_errors(self):
        """Test processing of Google API response with errors, using MAX_SAFE values."""
        mock_response = {
            'rows': [
                {'elements': [
                    {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}},
                    {'status': 'ZERO_RESULTS', 'error_message': 'No route found'}
                ]}
            ]
        }
        distance_matrix_list_km, time_matrix_list_min = DistanceMatrixBuilder._process_api_response(mock_response)

        # Check correct values for valid route (km and minutes)
        self.assertEqual(distance_matrix_list_km[0][0], 10.0) # 10km
        self.assertEqual(time_matrix_list_min[0][0], 10.0)   # 10 minutes

        # Check that MAX_SAFE_DISTANCE and MAX_SAFE_TIME are used for invalid routes
        self.assertEqual(distance_matrix_list_km[0][1], MAX_SAFE_DISTANCE) # MAX_SAFE_DISTANCE is in km
        self.assertEqual(time_matrix_list_min[0][1], MAX_SAFE_TIME)       # MAX_SAFE_TIME is in minutes

    @patch('requests.get')
    def test_send_request(self, mock_get):
        """Test sending requests to Google API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'status': 'OK', 'rows': []}
        mock_get.return_value = mock_response
        
        response = DistanceMatrixBuilder._send_request(
            ['Address 1'], 
            ['Address 2'], 
            'dummy_key'
        )
        
        # Check if request.get was called with the right parameters
        self.assertTrue(mock_get.called)
        args, kwargs = mock_get.call_args
        self.assertTrue('Address 1' in unquote(args[0]), f"Original address not found in URL: {args[0]}")
        self.assertTrue('Address 2' in unquote(args[0]), f"Destination address not found in URL: {args[0]}")
        self.assertTrue('key=dummy_key' in args[0])
        self.assertEqual(kwargs.get('timeout'), 10)
        
        # Check if response was properly processed
        self.assertEqual(response, {'status': 'OK', 'rows': []})

    @patch('time.sleep')
    @patch('requests.get')
    def test_send_request_with_retry(self, mock_get, mock_sleep):
        """Test sending request with retry logic."""
        # Mock a rate limit response followed by a success
        mock_error_response = MagicMock()
        mock_error_response.json.return_value = {
            'status': 'OVER_QUERY_LIMIT',
            'error_message': 'Rate limit exceeded'
        }
        
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = {
            'status': 'OK',
            'rows': []
        }
        
        # Return error on first call, success on second
        mock_get.side_effect = [mock_error_response, mock_success_response]
        
        response = DistanceMatrixBuilder._send_request_with_retry(
            ['Address 1'], 
            ['Address 2'], 
            'dummy_key'
        )
        
        # Verify retry logic was triggered
        self.assertEqual(mock_get.call_count, 2)
        self.assertTrue(mock_sleep.called)
        
        # Verify final response is the success response
        self.assertEqual(response, {'status': 'OK', 'rows': []})

    @patch('requests.get')
    def test_send_request_with_retry_max_retries(self, mock_get):
        """Test max retries being reached."""
        # Always return an error
        mock_error_response = MagicMock()
        mock_error_response.json.return_value = {
            'status': 'OVER_QUERY_LIMIT',
            'error_message': 'Rate limit exceeded'
        }
        
        mock_get.return_value = mock_error_response
        
        # Should raise an exception after MAX_RETRIES attempts
        with self.assertRaises(Exception) as context:
            DistanceMatrixBuilder._send_request_with_retry(
                ['Address 1'], 
                ['Address 2'], 
                'dummy_key'
            )
        
        self.assertTrue("All API request retries failed" in str(context.exception))

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder._send_request_with_retry')
    def test_fetch_distance_and_time_matrices(self, mock_send_retry):
        """Test fetching complete distance and time matrices in km and minutes."""
        mock_api_response_json = { # API raw response (meters, seconds)
            'status': 'OK',
            'rows': [
                {'elements': [
                    {'status': 'OK', 'distance': {'value': 0}, 'duration': {'value': 0}},
                    {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}}
                ]},
                {'elements': [
                    {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}},
                    {'status': 'OK', 'distance': {'value': 0}, 'duration': {'value': 0}}
                ]}
            ]
        }
        mock_send_retry.return_value = mock_api_response_json

        data_for_api = {
            "addresses": ["0.0,0.0", "1.0,1.0"], # Using lat,lon strings as _format_address does
            "API_key": "dummy_key"
        }
        # _fetch_distance_and_time_matrices internally calls _process_api_response
        # So, its output should be in km and minutes
        dist_list_km, time_list_min = DistanceMatrixBuilder._fetch_distance_and_time_matrices(data_for_api)

        expected_dist_km = [[0.0, 10.0], [10.0, 0.0]] # 10000m -> 10km
        expected_time_min = [[0.0, 10.0], [10.0, 0.0]] # 600s -> 10min

        self.assertEqual(dist_list_km, expected_dist_km)
        self.assertEqual(time_list_min, expected_time_min)
        mock_send_retry.assert_called_once()

    def test_sanitize_distance_matrix(self):
        """Test sanitization of distance matrix."""
        # Create a matrix with problematic values
        problematic_matrix = np.array([
            [0.0, 1.0, float('inf'), -1.0],
            [1.0, 0.0, np.nan, MAX_SAFE_DISTANCE * 2],
            [float('inf'), np.nan, 0.0, 5.0],
            [-1.0, MAX_SAFE_DISTANCE * 2, 5.0, 0.0]
        ])
        
        sanitized = self.builder._sanitize_distance_matrix(problematic_matrix)
        
        # Check infinity values are replaced
        self.assertFalse(np.isinf(sanitized).any())
        self.assertEqual(sanitized[0, 2], MAX_SAFE_DISTANCE)
        self.assertEqual(sanitized[2, 0], MAX_SAFE_DISTANCE)
        
        # Check NaN values are replaced
        self.assertFalse(np.isnan(sanitized).any())
        self.assertEqual(sanitized[1, 2], MAX_SAFE_DISTANCE)
        self.assertEqual(sanitized[2, 1], MAX_SAFE_DISTANCE)
        
        # Check negative values are replaced
        self.assertTrue((sanitized >= 0).all())
        self.assertEqual(sanitized[0, 3], 0)
        self.assertEqual(sanitized[3, 0], 0)
        
        # Check excessively large values are capped
        self.assertEqual(sanitized[1, 3], MAX_SAFE_DISTANCE)
        self.assertEqual(sanitized[3, 1], MAX_SAFE_DISTANCE)
        
        # Check valid values are left unchanged
        self.assertEqual(sanitized[0, 1], 1.0)
        self.assertEqual(sanitized[1, 0], 1.0)
        self.assertEqual(sanitized[2, 3], 5.0)
        self.assertEqual(sanitized[3, 2], 5.0)

    def test_apply_traffic_safely(self):
        """Test safe application of traffic factors."""
        # Create a base matrix
        base_matrix = np.array([
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 15.0],
            [20.0, 15.0, 0.0]
        ])
        
        # Create traffic data including valid and invalid values
        traffic_data = {
            (0, 1): 1.5,           # Valid factor
            (1, 2): 2.0,           # Valid factor
            (2, 0): 0.5,           # Below minimum (should use 1.0)
            (1, 0): 10.0,          # Above maximum (should be capped)
            (5, 5): 1.2            # Invalid indices (should be ignored)
        }
        
        result_matrix = self.builder._apply_traffic_safely(base_matrix, traffic_data)
        
        # Check valid factors applied correctly
        self.assertEqual(result_matrix[0, 1], 15.0)  # 10.0 * 1.5 = 15.0
        self.assertEqual(result_matrix[1, 2], 30.0)  # 15.0 * 2.0 = 30.0
        
        # Check factor below minimum is handled correctly
        self.assertEqual(result_matrix[2, 0], 20.0)  # Should not be reduced
        
        # Check factor above maximum is capped
        max_safe_factor = 5.0  # This is the value defined in the implementation
        self.assertEqual(result_matrix[1, 0], 10.0 * max_safe_factor)
        
        # Check invalid indices don't cause issues
        # This is implicitly tested by confirming the function runs without error

    @patch('route_optimizer.models.DistanceMatrixCache.objects.filter')
    def test_get_cached_matrix(self, mock_filter):
        """Test retrieving matrix from cache, including time matrix."""
        mock_cache_entry = MagicMock()
        mock_cache_entry.matrix_data = json.dumps([[0.0, 10.0], [10.0, 0.0]]) # Distances in km
        mock_cache_entry.time_matrix_data = json.dumps([[0.0, 5.0], [5.0, 0.0]]) # Times in minutes
        mock_cache_entry.location_ids = json.dumps(["loc1", "loc2"])

        mock_filter.return_value.first.return_value = mock_cache_entry

        cached_dist_matrix, cached_time_matrix, cached_loc_ids = DistanceMatrixBuilder.get_cached_matrix(self.locations[:2])

        self.assertTrue(np.array_equal(cached_dist_matrix, np.array([[0.0, 10.0], [10.0, 0.0]])))
        self.assertIsNotNone(cached_time_matrix)
        self.assertTrue(np.array_equal(cached_time_matrix, np.array([[0.0, 5.0], [5.0, 0.0]])))
        self.assertEqual(cached_loc_ids, ["loc1", "loc2"])
        mock_filter.assert_called_once()

    @patch('route_optimizer.models.DistanceMatrixCache.objects.filter')
    def test_get_cached_matrix_no_time_matrix(self, mock_filter):
        """Test retrieving matrix from cache when time_matrix_data is None."""
        mock_cache_entry = MagicMock()
        mock_cache_entry.matrix_data = json.dumps([[0.0, 10.0], [10.0, 0.0]])
        mock_cache_entry.time_matrix_data = None # Explicitly None
        mock_cache_entry.location_ids = json.dumps(["loc1", "loc2"])
        mock_filter.return_value.first.return_value = mock_cache_entry

        cached_dist_matrix, cached_time_matrix, cached_loc_ids = DistanceMatrixBuilder.get_cached_matrix(self.locations[:2])

        self.assertTrue(np.array_equal(cached_dist_matrix, np.array([[0.0, 10.0], [10.0, 0.0]])))
        self.assertIsNone(cached_time_matrix)
        self.assertEqual(cached_loc_ids, ["loc1", "loc2"])

    @patch('route_optimizer.models.DistanceMatrixCache.objects.update_or_create')
    def test_cache_matrix(self, mock_update_or_create):
        """Test caching a matrix."""
        # Test data
        distance_matrix = np.array([[0.0, 10.0], [10.0, 0.0]])
        location_ids = ["loc1", "loc2"]
        time_matrix = [[0, 600], [600, 0]]
        
        # Call cache_matrix
        DistanceMatrixBuilder.cache_matrix(distance_matrix, location_ids, time_matrix)
        
        # Verify update_or_create was called with the right arguments
        mock_update_or_create.assert_called_once()
        args, kwargs = mock_update_or_create.call_args
        
        # Check key arguments
        self.assertTrue('cache_key' in kwargs)
        self.assertTrue('defaults' in kwargs)
        
        # Check defaults
        defaults = kwargs['defaults']
        self.assertEqual(json.loads(defaults['matrix_data']), distance_matrix.tolist())
        self.assertEqual(json.loads(defaults['location_ids']), location_ids)
        self.assertEqual(json.loads(defaults['time_matrix_data']), time_matrix)
        self.assertTrue('created_at' in defaults)

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.get_cached_matrix')
    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.cache_matrix')
    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder._fetch_distance_and_time_matrices')
    def test_create_distance_matrix_from_api(self, mock_fetch_matrices, mock_cache_matrix, mock_get_cached):
        """Test API matrix creation (km and minutes) with cache miss."""
        mock_get_cached.return_value = None # Cache miss

        # _fetch_distance_and_time_matrices returns (dist_list_km, time_list_min)
        mock_dist_list_km = [[0.0, 10.0], [10.0, 0.0]] # Already in km
        mock_time_list_min = [[0.0, 5.0], [5.0, 0.0]]   # Already in minutes
        mock_fetch_matrices.return_value = (mock_dist_list_km, mock_time_list_min)

        dist_matrix_km, time_matrix_min, loc_ids = DistanceMatrixBuilder.create_distance_matrix_from_api(
            self.locations[:2], api_key='dummy_key', use_cache=True
        )

        self.assertEqual(dist_matrix_km.shape, (2, 2))
        self.assertEqual(time_matrix_min.shape, (2, 2))
        self.assertEqual(loc_ids, ["depot", "customer1"])

        self.assertAlmostEqual(dist_matrix_km[0, 1], 10.0) # km
        self.assertAlmostEqual(time_matrix_min[0, 1], 5.0) # minutes

        mock_get_cached.assert_called_once()
        mock_fetch_matrices.assert_called_once()
        # Check that the data passed to cache_matrix is what _fetch_matrices returned (np arrays)
        mock_cache_matrix.assert_called_once()
        args, _ = mock_cache_matrix.call_args
        self.assertTrue(np.array_equal(args[0], np.array(mock_dist_list_km))) # distance_matrix_km_np
        self.assertTrue(np.array_equal(args[2], np.array(mock_time_list_min))) # time_matrix_min_np

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.get_cached_matrix')
    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder._fetch_distance_and_time_matrices')
    def test_create_distance_matrix_from_api_with_cache_hit(self, mock_fetch_matrices, mock_get_cached):
        """Test API matrix creation with cache hit (km and minutes)."""
        mock_cached_dist_km = np.array([[0.0, 10.0], [10.0, 0.0]])   # km
        mock_cached_time_min = np.array([[0.0, 5.0], [5.0, 0.0]]) # minutes
        mock_cached_ids = ["depot", "customer1"]
        mock_get_cached.return_value = (mock_cached_dist_km, mock_cached_time_min, mock_cached_ids)

        dist_matrix_km, time_matrix_min, loc_ids = DistanceMatrixBuilder.create_distance_matrix_from_api(
            self.locations[:2], api_key='dummy_key', use_cache=True
        )

        self.assertTrue(np.array_equal(dist_matrix_km, mock_cached_dist_km))
        self.assertTrue(np.array_equal(time_matrix_min, mock_cached_time_min))
        self.assertEqual(loc_ids, mock_cached_ids)

        mock_get_cached.assert_called_once()
        mock_fetch_matrices.assert_not_called() # Should not fetch if cache hits

    def test_empty_locations(self):
        """Test handling of empty locations list for create_distance_matrix."""
        dist_matrix, time_matrix, location_ids = self.builder.create_distance_matrix(
            [], distance_calculation="haversine" # or any other
        )
        self.assertEqual(dist_matrix.shape, (0, 0))
        self.assertIsNotNone(time_matrix) # create_distance_matrix now returns an empty (0,0) array for time
        self.assertEqual(time_matrix.shape, (0,0))
        self.assertEqual(location_ids, [])

    def test_create_distance_matrix_api_fallback_no_key(self):
        """Test API fallback to Haversine if no API key is provided."""
        # We expect it to behave like create_distance_matrix with haversine
        # and no average_speed_kmh (so time_matrix is None)
        with patch.object(DistanceMatrixBuilder, 'create_distance_matrix_from_api') as mock_api_call:
            dist_matrix, time_matrix, loc_ids = self.builder.create_distance_matrix(
                self.locations,
                use_api=True, # Try to use API
                api_key=None  # But no key
            )
            # Verify that create_distance_matrix_from_api was NOT called due to api_key being None
            mock_api_call.assert_not_called() 

        # Based on create_distance_matrix executing its non-API path:
        # It uses Haversine (default) and average_speed_kmh is None.
        
        self.assertEqual(dist_matrix.shape, (4, 4))
        # time_matrix should be None because api_key was None (so API path not taken)
        # and average_speed_kmh was None (so local time estimation not done).
        self.assertIsNone(time_matrix) # Changed from assertIsNotNone
        
        # Check if Haversine distance was calculated as expected for the non-API path
        self.assertAlmostEqual(dist_matrix[0, 1], 157.2, delta=1.0) 
        self.assertEqual(loc_ids, ["depot", "customer1", "customer2", "customer3"])

        # The rest of the test (directly testing create_distance_matrix_from_api's fallback) remains valid:
        # Re-evaluating the fallback logic:
        # create_distance_matrix_from_api, if resolved_api_key is None, calls:
        # dist_matrix_fallback_km, time_matrix_fallback_min, loc_ids_fallback = DistanceMatrixBuilder.create_distance_matrix(
        #                                                         locations, use_haversine=True)
        # In create_distance_matrix, if average_speed_kmh is None, time_matrix_estimated_min is None.
        # So, the time_matrix returned from the fallback *should* be None.

        # Let's ensure the test matches this expectation.
        # We'll directly call create_distance_matrix_from_api with no key to test its fallback.
        dist_matrix_api_fallback, time_matrix_api_fallback, loc_ids_api_fallback = \
            DistanceMatrixBuilder.create_distance_matrix_from_api(
                self.locations, api_key=None, use_cache=False # No key, disable cache for direct test
            )

        self.assertEqual(dist_matrix_api_fallback.shape, (4, 4))
        self.assertAlmostEqual(dist_matrix_api_fallback[0, 1], 157.2, delta=1.0) # Haversine
        self.assertIsNone(time_matrix_api_fallback) # Fallback path in create_distance_matrix_from_api doesn't provide speed
        self.assertEqual(loc_ids_api_fallback, ["depot", "customer1", "customer2", "customer3"])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder._fetch_distance_and_time_matrices', side_effect=Exception("API Call Failed"))
    def test_create_distance_matrix_api_fallback_on_exception(self, mock_fetch):
        """Test API fallback to Haversine on general API exception."""
        dist_matrix_api_fallback, time_matrix_api_fallback, loc_ids_api_fallback = \
            DistanceMatrixBuilder.create_distance_matrix_from_api(
                self.locations, api_key="dummy_key", use_cache=False
            )
        
        mock_fetch.assert_called_once() # Ensure API fetch was attempted
        self.assertEqual(dist_matrix_api_fallback.shape, (4, 4))
        self.assertAlmostEqual(dist_matrix_api_fallback[0, 1], 157.2, delta=1.0) # Haversine
        self.assertIsNone(time_matrix_api_fallback) # Fallback path does not estimate time if not specified.
        self.assertEqual(loc_ids_api_fallback, ["depot", "customer1", "customer2", "customer3"])

    def test_distance_matrix_to_graph(self):
        """Test converting distance matrix to graph representation."""
        # Create a simple distance matrix
        distance_matrix = np.array([
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 15.0],
            [20.0, 15.0, 0.0]
        ])
        location_ids = ["loc1", "loc2", "loc3"]
        
        # Convert to graph
        graph = DistanceMatrixBuilder.distance_matrix_to_graph(distance_matrix, location_ids)
        
        # Check graph structure
        self.assertEqual(len(graph), 3)
        self.assertEqual(len(graph["loc1"]), 2)  # Two connections from loc1
        
        # Check specific distances
        self.assertEqual(graph["loc1"]["loc2"], 10.0)
        self.assertEqual(graph["loc1"]["loc3"], 20.0)
        self.assertEqual(graph["loc2"]["loc3"], 15.0)
        
        # Check no self-connections
        self.assertNotIn("loc1", graph["loc1"])
        self.assertNotIn("loc2", graph["loc2"])
        self.assertNotIn("loc3", graph["loc3"])

if __name__ == '__main__':
    unittest.main()
