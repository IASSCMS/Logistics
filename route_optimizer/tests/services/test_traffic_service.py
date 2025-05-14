from django.test import TestCase
import numpy as np
from unittest.mock import patch, MagicMock

from route_optimizer.services.traffic_service import TrafficService
from route_optimizer.core.types_1 import Location
from route_optimizer.core.distance_matrix import DistanceMatrixBuilder, MAX_SAFE_DISTANCE 

class TrafficServiceTest(TestCase):
    def setUp(self):
        self.locations = [
            Location(id="loc1", name="Location 1", latitude=0.0, longitude=0.0),
            Location(id="loc2", name="Location 2", latitude=1.0, longitude=1.0),
            Location(id="loc3", name="Location 3", latitude=2.0, longitude=0.0),
        ]
        self.location_ids = [loc.id for loc in self.locations]

    def test_apply_traffic_factors(self):
        """Test applying traffic factors delegates to DistanceMatrixBuilder."""
        matrix = np.array([[0.0, 10.0], [10.0, 0.0]], dtype=float)
        traffic_data = {(0, 1): 1.5, (1, 0): 2.0}
        
        expected_matrix = np.array([[0.0, 15.0], [20.0, 0.0]], dtype=float)

        # Mock DistanceMatrixBuilder.add_traffic_factors to ensure it's called
        with patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.add_traffic_factors') as mock_add_factors:
            mock_add_factors.return_value = expected_matrix # Simulate the behavior of the real method
            
            adjusted_matrix = TrafficService.apply_traffic_factors(matrix.copy(), traffic_data)
            
            mock_add_factors.assert_called_once()
            # Check that the first argument to mock_add_factors is a copy of the original matrix
            # and the second is the traffic_data. np.array_equal for the matrix.
            self.assertTrue(np.array_equal(mock_add_factors.call_args[0][0], matrix))
            self.assertEqual(mock_add_factors.call_args[0][1], traffic_data)
            
            # Also verify the result from the mock
            self.assertTrue(np.array_equal(adjusted_matrix, expected_matrix))

    def test_calculate_distance_haversine_internal(self):
        """Test the internal _calculate_distance_haversine method."""
        service = TrafficService()
        loc1 = Location(id="A", latitude=0.0, longitude=0.0)
        loc2 = Location(id="B", latitude=1.0, longitude=1.0)
        
        # Mocking the static _haversine_distance from DistanceMatrixBuilder
        with patch.object(DistanceMatrixBuilder, '_haversine_distance', return_value=157.2) as mock_haversine:
            dist = service._calculate_distance_haversine(loc1, loc2)
            self.assertAlmostEqual(dist, 157.2, delta=0.1)
            mock_haversine.assert_called_once_with(0.0, 0.0, 1.0, 1.0)

    def test_calculate_distance_haversine_missing_coords(self):
        """Test _calculate_distance_haversine with missing coordinates."""
        service = TrafficService()
        loc1 = Location(id="A", latitude=0.0, longitude=None) # Missing longitude
        loc2 = Location(id="B", latitude=1.0, longitude=1.0)
        
        dist = service._calculate_distance_haversine(loc1, loc2)
        self.assertEqual(dist, float('inf')) # Expecting inf for error

        loc3 = Location(id="C", latitude=None, longitude=0.0) # Missing latitude
        dist2 = service._calculate_distance_haversine(loc3, loc2)
        self.assertEqual(dist2, float('inf'))


    def test_create_road_graph_no_api_key(self):
        """Test create_road_graph fallback to Haversine when no API key is provided."""
        service = TrafficService(api_key=None)
        
        # Mock _calculate_distance_haversine to control its output
        with patch.object(service, '_calculate_distance_haversine') as mock_calc_dist:
            # Make loc1 <-> loc2 = 10km, loc1 <-> loc3 = 20km, loc2 <-> loc3 = 15km
            def side_effect_calc_dist(l1, l2):
                if (l1.id == "loc1" and l2.id == "loc2") or (l1.id == "loc2" and l2.id == "loc1"): return 10.0
                if (l1.id == "loc1" and l2.id == "loc3") or (l1.id == "loc3" and l2.id == "loc1"): return 20.0
                if (l1.id == "loc2" and l2.id == "loc3") or (l1.id == "loc3" and l2.id == "loc2"): return 15.0
                return MAX_SAFE_DISTANCE
            mock_calc_dist.side_effect = side_effect_calc_dist
            
            graph = service.create_road_graph(self.locations)

            self.assertEqual(len(graph['nodes']), 3)
            self.assertIn("loc1", graph['nodes'])
            self.assertEqual(len(graph['edges']), 3)
            self.assertIn("loc1", graph['edges'])
            
            # Check distances and structure
            self.assertEqual(graph['edges']['loc1']['loc2']['distance'], 10.0)
            self.assertIsNone(graph['edges']['loc1']['loc2']['time']) # No time with Haversine fallback
            self.assertEqual(graph['edges']['loc1']['loc3']['distance'], 20.0)
            self.assertEqual(graph['edges']['loc2']['loc3']['distance'], 15.0)
            self.assertNotIn("loc1", graph['edges']['loc1'].get("loc1", {})) # No self-loops
            self.assertEqual(mock_calc_dist.call_count, 6) # Each pair bidirectionally: 3*2=6

    def test_create_road_graph_empty_locations(self):
        """Test create_road_graph with an empty list of locations."""
        service = TrafficService(api_key=None)
        graph = service.create_road_graph([])
        self.assertEqual(graph, {'nodes': {}, 'edges': {}})

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix_from_api')
    def test_create_road_graph_with_api_success(self, mock_create_matrix_api):
        """Test create_road_graph with a successful API call."""
        service = TrafficService(api_key="dummy_key")
        
        # Mock API response (distance_km, time_minutes, location_ids)
        mock_dist_km = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]], dtype=float)
        mock_time_min = np.array([[0, 5, 10], [5, 0, 8], [10, 8, 0]], dtype=float) # Times in minutes
        mock_create_matrix_api.return_value = (mock_dist_km, mock_time_min, self.location_ids)
        
        graph = service.create_road_graph(self.locations)
        
        mock_create_matrix_api.assert_called_once_with(self.locations, "dummy_key", use_cache=True)
        
        self.assertEqual(graph['edges']['loc1']['loc2']['distance'], 10.0)
        self.assertEqual(graph['edges']['loc1']['loc2']['time'], 5.0) # Should be minutes
        self.assertEqual(graph['edges']['loc1']['loc3']['distance'], 20.0)
        self.assertEqual(graph['edges']['loc1']['loc3']['time'], 10.0) # Should be minutes
        self.assertEqual(graph['edges']['loc2']['loc3']['distance'], 15.0)
        self.assertEqual(graph['edges']['loc2']['loc3']['time'], 8.0) # Should be minutes

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix_from_api')
    def test_create_road_graph_with_api_time_matrix_none(self, mock_create_matrix_api):
        """Test create_road_graph when API returns None for time matrix."""
        service = TrafficService(api_key="dummy_key")
        
        mock_dist_km = np.array([[0, 10], [10, 0]], dtype=float)
        mock_create_matrix_api.return_value = (mock_dist_km, None, self.location_ids[:2]) # Time matrix is None
        
        graph = service.create_road_graph(self.locations[:2]) # Use only 2 locations for this test
        
        self.assertEqual(graph['edges']['loc1']['loc2']['distance'], 10.0)
        self.assertIsNone(graph['edges']['loc1']['loc2']['time'])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix_from_api', side_effect=Exception("API Network Error"))
    @patch.object(TrafficService, '_calculate_distance_haversine') # Also mock this for the fallback
    def test_create_road_graph_api_failure_fallback(self, mock_calc_haversine, mock_create_matrix_api):
        """Test create_road_graph fallback to Haversine on API exception."""
        service = TrafficService(api_key="dummy_key")
        
        # Setup mock Haversine for fallback
        mock_calc_haversine.return_value = 25.0 
        
        graph = service.create_road_graph(self.locations)
        
        mock_create_matrix_api.assert_called_once_with(self.locations, "dummy_key", use_cache=True)
        self.assertTrue(mock_calc_haversine.called) # Check Haversine was used
        
        # Check one of the distances to confirm fallback logic applied
        self.assertEqual(graph['edges']['loc1']['loc2']['distance'], 25.0) 
        self.assertIsNone(graph['edges']['loc1']['loc2']['time'])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix_from_api')
    def test_create_road_graph_api_location_id_mismatch(self, mock_create_matrix_api):
        """Test create_road_graph handles API location ID mismatch by raising ValueError."""
        service = TrafficService(api_key="dummy_key")
        
        mismatched_ids = ["loc1_api", "loc2_api", "loc3_api"]
        mock_dist_km = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]], dtype=float)
        mock_time_min = np.array([[0, 5, 10], [5, 0, 8], [10, 8, 0]], dtype=float)
        mock_create_matrix_api.return_value = (mock_dist_km, mock_time_min, mismatched_ids)
        
        # The service's create_road_graph should catch the ValueError from ID mismatch and then fallback
        # Let's verify the fallback behavior.
        with patch.object(service, '_calculate_distance_haversine') as mock_calc_dist_fallback:
            mock_calc_dist_fallback.return_value = 33.0 # Arbitrary fallback distance
            
            # Since the ID mismatch triggers an exception that's caught and leads to fallback:
            graph = service.create_road_graph(self.locations)

            mock_create_matrix_api.assert_called_once()
            self.assertTrue(mock_calc_dist_fallback.called) # Fallback was triggered
            self.assertEqual(graph['edges']['loc1']['loc2']['distance'], 33.0) # Verify fallback value