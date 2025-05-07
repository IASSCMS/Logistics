import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import requests
import json

from route_optimizer.core.distance_matrix import DistanceMatrixBuilder, Location

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

    @patch('requests.get')
    def test_create_distance_matrix_google(self, mock_get):
        """Test creating a distance matrix using Google Maps API."""
        # Mock the Google Maps API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'OK',
            'rows': [
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 0}},
                        {'status': 'OK', 'distance': {'value': 10000}},
                        {'status': 'OK', 'distance': {'value': 20000}},
                        {'status': 'OK', 'distance': {'value': 30000}}
                    ]
                },
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 10000}},
                        {'status': 'OK', 'distance': {'value': 0}},
                        {'status': 'OK', 'distance': {'value': 15000}},
                        {'status': 'OK', 'distance': {'value': 25000}}
                    ]
                },
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 20000}},
                        {'status': 'OK', 'distance': {'value': 15000}},
                        {'status': 'OK', 'distance': {'value': 0}},
                        {'status': 'OK', 'distance': {'value': 10000}}
                    ]
                },
                {
                    'elements': [
                        {'status': 'OK', 'distance': {'value': 30000}},
                        {'status': 'OK', 'distance': {'value': 25000}},
                        {'status': 'OK', 'distance': {'value': 10000}},
                        {'status': 'OK', 'distance': {'value': 0}}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        with patch.object(self.builder, '_get_api_key', return_value='dummy_key'):
            matrix, location_ids = self.builder.create_distance_matrix(
                self.locations, 
                distance_calculation="google"
            )
            
            # Check matrix shape
            self.assertEqual(matrix.shape, (4, 4))
            
            # Check location IDs
            self.assertEqual(location_ids, ["depot", "customer1", "customer2", "customer3"])
            
            # Check some specific distances (converted to km)
            self.assertEqual(matrix[0, 1], 10.0)  # 10000 meters = 10 km
            self.assertEqual(matrix[1, 2], 15.0)  # 15000 meters = 15 km
            
            # Check diagonal (should be zeros)
            for i in range(4):
                self.assertEqual(matrix[i, i], 0.0)

    @patch('requests.get')
    def test_google_api_error_fallback(self, mock_get):
        """Test fallback to Haversine when Google API fails."""
        # Mock a failed API response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_get.return_value = mock_response
        
        with patch.object(self.builder, '_get_api_key', return_value='dummy_key'):
            with patch.object(self.builder, '_haversine_distance', return_value=10.0):
                matrix, location_ids = self.builder.create_distance_matrix(
                    self.locations, 
                    distance_calculation="google"
                )
                
                # Should have fallen back to Haversine
                self.assertEqual(matrix.shape, (4, 4))
                # All non-diagonal entries should be 10.0 due to our mock
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            self.assertEqual(matrix[i, j], 10.0)

    def test_add_traffic_factors(self):
        """Test adding traffic factors to a distance matrix."""
        # Create a simple distance matrix
        base_matrix = np.array([
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 15.0],
            [20.0, 15.0, 0.0]
        ])
        
        # Define traffic factors for specific node pairs
        traffic_data = {
            (0, 1): 1.5,  # 50% more time from node 0 to 1
            (1, 2): 2.0   # Twice as long from node 1 to 2
        }
        
        # Apply traffic factors
        result_matrix = self.builder.add_traffic_factors(base_matrix, traffic_data)
        
        # Check that traffic factors were applied correctly
        self.assertEqual(result_matrix[0, 1], 15.0)  # 10.0 * 1.5 = 15.0
        self.assertEqual(result_matrix[1, 2], 30.0)  # 15.0 * 2.0 = 30.0
        
        # Check that other values remain unchanged
        self.assertEqual(result_matrix[0, 2], 20.0)
        self.assertEqual(result_matrix[1, 0], 10.0)
        self.assertEqual(result_matrix[2, 0], 20.0)
        self.assertEqual(result_matrix[2, 1], 15.0)

    def test_empty_locations(self):
        """Test handling of empty locations list."""
        matrix, location_ids = self.builder.create_distance_matrix([])
        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(location_ids, [])

if __name__ == '__main__':
    unittest.main()
