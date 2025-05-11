import unittest
from unittest.mock import patch, MagicMock
from urllib.parse import unquote
import numpy as np
import requests
import json
from datetime import datetime, timedelta

from route_optimizer.core.distance_matrix import DistanceMatrixBuilder, Location
from route_optimizer.core.constants import DISTANCE_SCALING_FACTOR, MAX_SAFE_DISTANCE

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
        matrix, location_ids = self.builder.create_distance_matrix(
            self.locations, 
            distance_calculation="euclidean"
        )
        
        # Check matrix shape
        self.assertEqual(matrix.shape, (4, 4))
        
        # Check location IDs
        self.assertEqual(location_ids, ["depot", "customer1", "customer2", "customer3"])
        
        # Check some specific distances (Euclidean)
        # Depot to Customer1 (0,0) to (1,1) = sqrt(2) ≈ 1.414
        self.assertAlmostEqual(matrix[0, 1], 1.414, delta=0.001)
        
        # Check diagonal (should be zeros)
        for i in range(4):
            self.assertEqual(matrix[i, i], 0.0)

    def test_create_distance_matrix_haversine(self):
        """Test creating a distance matrix using Haversine distance."""
        matrix, location_ids = self.builder.create_distance_matrix(
            self.locations, 
            distance_calculation="haversine"
        )
        
        # Check matrix shape
        self.assertEqual(matrix.shape, (4, 4))
        
        # Check location IDs
        self.assertEqual(location_ids, ["depot", "customer1", "customer2", "customer3"])
        
        # Check diagonal (should be zeros)
        for i in range(4):
            self.assertEqual(matrix[i, i], 0.0)

    def test_process_api_response(self):
        """Test processing of Google API response data."""
        mock_response = {
            'rows': [
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}},
                        {'status': 'OK', 'distance': {'value': 20000}, 'duration': {'value': 1200}}
                    ]
                },
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 30000}, 'duration': {'value': 1800}},
                        {'status': 'OK', 'distance': {'value': 5000}, 'duration': {'value': 300}}
                    ]
                }
            ]
        }
        
        distance_matrix, time_matrix = DistanceMatrixBuilder._process_api_response(mock_response)
        
        # Check that distances are correctly converted to kilometers
        self.assertEqual(distance_matrix[0][0], 10.0)  # 10000m = 10km
        self.assertEqual(distance_matrix[0][1], 20.0)  # 20000m = 20km
        self.assertEqual(distance_matrix[1][0], 30.0)  # 30000m = 30km
        self.assertEqual(distance_matrix[1][1], 5.0)   # 5000m = 5km
        
        # Check that times are correctly processed (in seconds)
        self.assertEqual(time_matrix[0][0], 600)
        self.assertEqual(time_matrix[0][1], 1200)
        self.assertEqual(time_matrix[1][0], 1800)
        self.assertEqual(time_matrix[1][1], 300)

    def test_process_api_response_with_errors(self):
        """Test processing of Google API response with errors."""
        mock_response = {
            'rows': [
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}},
                        {'status': 'ZERO_RESULTS', 'error_message': 'No route found'}
                    ]
                }
            ]
        }
        
        distance_matrix, time_matrix = DistanceMatrixBuilder._process_api_response(mock_response)
        
        # Check correct values for valid route
        self.assertEqual(distance_matrix[0][0], 10.0)
        self.assertEqual(time_matrix[0][0], 600)
        
        # Check that inf is used for invalid routes
        self.assertEqual(distance_matrix[0][1], float('inf'))
        self.assertEqual(time_matrix[0][1], float('inf'))

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

    @patch('requests.get')
    def test_fetch_distance_and_time_matrices(self, mock_get):
        """Test fetching complete distance and time matrices."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'status': 'OK',
            'rows': [
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 0}, 'duration': {'value': 0}},
                        {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}}
                    ]
                },
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}},
                        {'status': 'OK', 'distance': {'value': 0}, 'duration': {'value': 0}}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Patch the _send_request_with_retry to use our mock
        with patch.object(DistanceMatrixBuilder, '_send_request_with_retry', return_value=mock_response.json()):
            data = {
                "addresses": ["Address 1", "Address 2"],
                "API_key": "dummy_key"
            }
            distance_matrix, time_matrix = DistanceMatrixBuilder._fetch_distance_and_time_matrices(data)
            
            # Should have 2 rows in the result
            self.assertEqual(len(distance_matrix), 2)
            self.assertEqual(len(time_matrix), 2)
            
            # Check values
            self.assertEqual(distance_matrix[0][0], 0.0)
            self.assertEqual(distance_matrix[0][1], 10.0)
            self.assertEqual(distance_matrix[1][0], 10.0)
            self.assertEqual(distance_matrix[1][1], 0.0)
            
            self.assertEqual(time_matrix[0][0], 0)
            self.assertEqual(time_matrix[0][1], 600)
            self.assertEqual(time_matrix[1][0], 600)
            self.assertEqual(time_matrix[1][1], 0)

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
        """Test retrieving matrix from cache."""
        # Create a mock cached result
        mock_cache = MagicMock()
        mock_cache.matrix_data = json.dumps([[0.0, 10.0], [10.0, 0.0]])
        mock_cache.location_ids = json.dumps(["loc1", "loc2"])
        
        # Make filter return our mock cache object
        mock_filter.return_value.first.return_value = mock_cache
        
        # Create test locations
        locations = [
            Location(id="loc1", name="Location 1", latitude=0.0, longitude=0.0),
            Location(id="loc2", name="Location 2", latitude=1.0, longitude=1.0)
        ]
        
        # Get cached matrix
        matrix, ids = DistanceMatrixBuilder.get_cached_matrix(locations)
        
        # Check result
        self.assertEqual(matrix.tolist(), [[0.0, 10.0], [10.0, 0.0]])
        self.assertEqual(ids, ["loc1", "loc2"])
        
        # Verify the filter was called with the right arguments
        mock_filter.assert_called_once()
        # We can't easily verify the exact arguments due to the hash being computed

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

    @patch('requests.get')
    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.get_cached_matrix')
    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.cache_matrix')
    def test_create_distance_matrix_from_api(self, mock_cache_matrix, mock_get_cached, mock_get):
        """Test full end-to-end API matrix creation."""
        # Mock cached matrix to return None (cache miss)
        mock_get_cached.return_value = None
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'status': 'OK',
            'rows': [
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 0}, 'duration': {'value': 0}},
                        {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}}
                    ]
                },
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 10000}, 'duration': {'value': 600}},
                        {'status': 'OK', 'distance': {'value': 0}, 'duration': {'value': 0}}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Patch the helper methods to use our mocks
        with patch.object(DistanceMatrixBuilder, '_send_request_with_retry', return_value=mock_response.json()):
            # Call the method
            matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix_from_api(
                self.locations[:2],  # Just use two locations
                api_key='dummy_key',
                use_cache=True
            )
            
            # Verify the matrix
            self.assertEqual(matrix.shape, (2, 2))
            self.assertEqual(matrix[0, 1], 10.0)  # 10km
            self.assertEqual(matrix[1, 0], 10.0)  # 10km
            
            # Verify cache was checked and result was cached
            mock_get_cached.assert_called_once()
            mock_cache_matrix.assert_called_once()

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.get_cached_matrix')
    def test_create_distance_matrix_from_api_with_cache_hit(self, mock_get_cached):
        """Test API matrix creation with cache hit."""
        # Mock cached result
        mock_cached_matrix = np.array([[0.0, 10.0], [10.0, 0.0]])
        mock_cached_ids = ["depot", "customer1"]
        mock_get_cached.return_value = (mock_cached_matrix, mock_cached_ids)
        
        # Call the method
        matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix_from_api(
            self.locations[:2],  # Just use two locations
            api_key='dummy_key',
            use_cache=True
        )
        
        # Verify the result is from cache
        self.assertTrue(np.array_equal(matrix, mock_cached_matrix))
        self.assertEqual(location_ids, mock_cached_ids)
        
        # Verify cache was checked
        mock_get_cached.assert_called_once()

    def test_empty_locations(self):
        """Test handling of empty locations list."""
        matrix, location_ids = self.builder.create_distance_matrix([])
        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(location_ids, [])

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
