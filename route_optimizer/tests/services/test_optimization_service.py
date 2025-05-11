"""
Tests for the optimization service.

This module contains comprehensive tests for the OptimizationService class.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.core.types_1 import Location, OptimizationResult
from route_optimizer.models import Vehicle, Delivery


class TestOptimizationService(unittest.TestCase):
    """Test cases for OptimizationService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock VRP solver and pathfinder
        self.mock_vrp_solver = MagicMock()
        self.mock_path_finder = MagicMock()
        
        # Initialize service with mocks
        self.service = OptimizationService(
            vrp_solver=self.mock_vrp_solver,
            path_finder=self.mock_path_finder
        )
        
        # Sample locations
        self.locations = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True),
            Location(id="customer1", name="Customer 1", latitude=1.0, longitude=0.0),
            Location(id="customer2", name="Customer 2", latitude=0.0, longitude=1.0),
            Location(id="customer3", name="Customer 3", latitude=1.0, longitude=1.0)
        ]
        
        # Sample vehicles
        self.vehicles = [
            Vehicle(
                id="vehicle1",
                capacity=10.0,
                fixed_cost=100.0,
                cost_per_km=2.0,
                start_location_id="depot",
                end_location_id="depot"
            ),
            Vehicle(
                id="vehicle2",
                capacity=15.0,
                fixed_cost=150.0,
                cost_per_km=2.5,
                start_location_id="depot",
                end_location_id="depot"
            )
        ]
        
        # Sample deliveries
        self.deliveries = [
            Delivery(id="delivery1", location_id="customer1", demand=5.0),
            Delivery(id="delivery2", location_id="customer2", demand=3.0),
            Delivery(id="delivery3", location_id="customer3", demand=6.0)
        ]
        
        # Mock distance matrix and location IDs
        self.distance_matrix = np.array([
            [0.0, 1.0, 1.0, 1.4],
            [1.0, 0.0, 1.4, 1.0],
            [1.0, 1.4, 0.0, 1.0],
            [1.4, 1.0, 1.0, 0.0]
        ])
        self.location_ids = ["depot", "customer1", "customer2", "customer3"]
        
        # Sample graph for pathfinding tests
        self.graph = {
            "matrix": self.distance_matrix,
            "location_ids": self.location_ids
        }
        
        # Define MAX_SAFE_DISTANCE for testing sanitize method
        global MAX_SAFE_DISTANCE
        MAX_SAFE_DISTANCE = 1000.0

    # --- Basic Optimization Tests ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    def test_optimize_routes_basic(self, mock_get_depot, mock_create_matrix):
        """Test basic route optimization without traffic or time windows."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        
        # Set up VRP solver mock
        self.mock_vrp_solver.solve.return_value = OptimizationResult(
            status='success',
            routes=[[0, 1, 2, 0], [0, 3, 0]],
            total_distance=6.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        # Call the service
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        
        # Verify the result
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.total_distance, 6.0)
        self.assertEqual(len(result.routes), 2)
        self.assertEqual(len(result.unassigned_deliveries), 0)
        
        # Verify the mocks were called correctly
        mock_create_matrix.assert_called_once()
        self.mock_vrp_solver.solve.assert_called_once()
        mock_get_depot.assert_called_once()

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    def test_optimize_routes_with_traffic(self, mock_get_depot, mock_create_matrix):
        """Test route optimization with traffic data."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        
        # Set up VRP solver mock
        self.mock_vrp_solver.solve.return_value = OptimizationResult(
            status='success',
            routes=[[0, 1, 2, 0], [0, 3, 0]],
            total_distance=8.0,  # Increased due to traffic
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        # Sample traffic data
        traffic_data = {(0, 1): 1.5, (1, 2): 1.2}
        
        # Call the service with patched _apply_traffic_safely
        with patch.object(self.service, '_apply_traffic_safely', return_value=self.distance_matrix):
            result = self.service.optimize_routes(
                locations=self.locations,
                vehicles=self.vehicles,
                deliveries=self.deliveries,
                consider_traffic=True,
                traffic_data=traffic_data
            )
        
        # Verify the result
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.total_distance, 8.0)  # Should be increased from traffic
        
        # Verify the mocks were called correctly
        mock_create_matrix.assert_called_once()
        self.mock_vrp_solver.solve.assert_called_once()

    # --- Time Windows Test ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    def test_optimize_routes_with_time_windows(self, mock_get_depot, mock_create_matrix):
        """Test route optimization with time windows."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        
        # Set up VRP solver mock
        self.mock_vrp_solver.solve_with_time_windows.return_value = OptimizationResult(
            status='success',
            routes=[[0, 1, 2, 0], [0, 3, 0]],
            total_distance=6.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        # Call the service
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            consider_time_windows=True
        )
        
        # Verify the result
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.total_distance, 6.0)
        
        # Verify the solve_with_time_windows method was called
        self.mock_vrp_solver.solve_with_time_windows.assert_called_once()
        self.mock_vrp_solver.solve.assert_not_called()

    # --- Edge Case Tests ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    def test_validation_errors(self, mock_get_depot, mock_create_matrix):
        """Test validation errors are handled correctly."""
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        
        # Test with invalid location (missing coordinates)
        invalid_locations = [
            Location(id="invalid", name="Invalid", is_depot=False)  # Missing lat/long
        ]
        
        result = self.service.optimize_routes(
            locations=invalid_locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        
        self.assertEqual(result.status, 'error')
        self.assertIn('error', result.statistics)
        self.assertIn('latitude', result.statistics['error'].lower())

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    def test_exception_handling(self, mock_get_depot, mock_create_matrix):
        """Should handle exceptions gracefully."""
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        self.mock_vrp_solver.solve.side_effect = Exception("Test exception")
        
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        
        self.assertEqual(result.status, 'error')
        self.assertEqual(len(result.routes), 0)
        self.assertEqual(len(result.unassigned_deliveries), 3)  # All deliveries unassigned
        self.assertIn('error', result.statistics)
        self.assertIn('Test exception', result.statistics['error'])

    # --- Helper Method Tests ---

    def test_sanitize_distance_matrix(self):
        """Test sanitizing distance matrix."""
        # Create a matrix with problematic values
        matrix = np.array([
            [0.0, 1.0, float('inf'), -5.0],
            [1.0, 0.0, float('nan'), 2.0],
            [float('inf'), float('nan'), 0.0, 5000.0],
            [-5.0, 2.0, 5000.0, 0.0]
        ])
        
        # Call the sanitize method
        result = self.service._sanitize_distance_matrix(matrix)
        
        # Check that infinities were replaced with MAX_SAFE_DISTANCE
        self.assertEqual(result[0, 2], MAX_SAFE_DISTANCE)
        self.assertEqual(result[2, 0], MAX_SAFE_DISTANCE)
        
        # Check that NaNs were replaced with MAX_SAFE_DISTANCE
        self.assertEqual(result[1, 2], MAX_SAFE_DISTANCE)
        self.assertEqual(result[2, 1], MAX_SAFE_DISTANCE)
        
        # Check that negative values were replaced with 0
        self.assertEqual(result[0, 3], 0.0)
        self.assertEqual(result[3, 0], 0.0)
        
        # Check that values exceeding MAX_SAFE_DISTANCE were capped
        self.assertEqual(result[2, 3], MAX_SAFE_DISTANCE)
        self.assertEqual(result[3, 2], MAX_SAFE_DISTANCE)

    def test_apply_traffic_safely(self):
        """Test applying traffic factors safely."""
        # Create a simple matrix
        matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0]
        ])
        
        # Define traffic factors
        traffic_data = {
            (0, 1): 1.5,    # Normal factor
            (1, 2): 10.0,   # Excessive factor (should be capped)
            (2, 0): -1.0,   # Invalid factor (should be set to minimum 1.0)
            (5, 5): 2.0     # Out of bounds index (should be ignored)
        }
        
        # Apply traffic factors
        result = self.service._apply_traffic_safely(matrix, traffic_data)
        
        # Check normal factor was applied
        self.assertEqual(result[0, 1], 1.5)  # 1.0 * 1.5
        
        # Check excessive factor was capped (assuming max_safe_factor=5.0)
        self.assertEqual(result[1, 2], 15.0 if 10.0 <= 5.0 else 3.0 * 5.0)
        
        # Check invalid factor was set to minimum 1.0
        self.assertEqual(result[2, 0], 2.0)  # Unchanged because factor < 1.0
        
        # Check out of bounds index was ignored
        self.assertEqual(result[0, 0], 0.0)  # Unchanged

    def test_convert_to_optimization_result(self):
        """Test converting dictionary to OptimizationResult."""
        # Create a sample result dictionary
        result_dict = {
            'status': 'success',
            'routes': [[0, 1, 0], [0, 2, 0]],
            'total_distance': 5.0,
            'total_cost': 150.0,
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1},
            'unassigned_deliveries': ['delivery3'],
            'detailed_routes': [],
            'statistics': {'total_time': 120}
        }
        
        # Convert to OptimizationResult
        result = self.service._convert_to_optimization_result(result_dict)
        
        # Verify the conversion
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.routes, [[0, 1, 0], [0, 2, 0]])
        self.assertEqual(result.total_distance, 5.0)
        self.assertEqual(result.total_cost, 150.0)
        self.assertEqual(result.assigned_vehicles, {'vehicle1': 0, 'vehicle2': 1})
        self.assertEqual(result.unassigned_deliveries, ['delivery3'])
        self.assertEqual(result.detailed_routes, [])
        self.assertEqual(result.statistics, {'total_time': 120})

    def test_convert_empty_result(self):
        """Test converting an empty or invalid result dictionary."""
        # Test with empty dictionary
        result = self.service._convert_to_optimization_result({})
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(result.routes, [])
        self.assertEqual(result.total_distance, 0.0)
        
        # Test with None
        result = self.service._convert_to_optimization_result(None)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.status, 'error')
        self.assertIn('error', result.statistics)
        self.assertIn('Conversion error', result.statistics['error'])

    # --- External API Tests ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    @patch('route_optimizer.services.traffic_service.TrafficService.create_road_graph')
    def test_optimize_routes_with_api(self, mock_create_graph, mock_get_depot, mock_create_matrix):
        """Test optimization using external API."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        mock_create_graph.return_value = self.graph
        
        # Set up VRP solver mock
        self.mock_vrp_solver.solve.return_value = OptimizationResult(
            status='success',
            routes=[[0, 1, 2, 0], [0, 3, 0]],
            total_distance=6.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        # Call the service with use_api=True
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            use_api=True,
            api_key='test_api_key'
        )
        
        # Verify the result
        self.assertEqual(result.status, 'success')
        
        # Verify the API was used
        mock_create_matrix.assert_called_once_with(
            self.locations, use_api=True, api_key='test_api_key'
        )
        mock_create_graph.assert_called_once()

    # --- Add Detailed Paths Tests ---

    def test_add_detailed_paths_optimization_result(self):
        """Test adding detailed paths to OptimizationResult."""
        # Create a sample optimization result
        result = OptimizationResult(
            status='success',
            routes=[[0, 1, 0], [0, 2, 0]],
            total_distance=4.0,
            total_cost=100.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        # Mock path annotator
        mock_annotator = MagicMock()
        
        # Patch PathAnnotator constructor
        with patch('route_optimizer.services.path_annotation_service.PathAnnotator', return_value=mock_annotator):
            # Call the method to add detailed paths
            self.service._add_detailed_paths(
                result, 
                self.graph, 
                self.location_ids
            )
        
        # Verify that detailed_routes were initialized
        self.assertTrue(hasattr(result, 'detailed_routes'))
        self.assertEqual(len(result.detailed_routes), 2)  # Two routes
        
        # Verify the vehicle assignments
        self.assertEqual(result.detailed_routes[0]['vehicle_id'], 'vehicle1')
        self.assertEqual(result.detailed_routes[1]['vehicle_id'], 'vehicle2')
        
        # Verify the annotator was called
        mock_annotator.annotate.assert_called_once_with(result, self.graph)

    def test_add_detailed_paths_dict(self):
        """Test adding detailed paths to result dictionary."""
        # Create a sample result dictionary
        result = {
            'status': 'success',
            'routes': [[0, 1, 0], [0, 2, 0]],
            'total_distance': 4.0,
            'total_cost': 100.0,
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1},
            'unassigned_deliveries': [],
            'detailed_routes': [],
            'statistics': {}
        }
        
        # Mock path annotator
        mock_annotator = MagicMock()
        
        # Patch PathAnnotator constructor
        with patch('route_optimizer.services.path_annotation_service.PathAnnotator', return_value=mock_annotator):
            # Call the method to add detailed paths
            self.service._add_detailed_paths(
                result, 
                self.graph, 
                self.location_ids
            )
        
        # Verify that detailed_routes were initialized
        self.assertIn('detailed_routes', result)
        self.assertEqual(len(result['detailed_routes']), 2)  # Two routes
        
        # Verify the vehicle assignments
        self.assertEqual(result['detailed_routes'][0]['vehicle_id'], 'vehicle1')
        self.assertEqual(result['detailed_routes'][1]['vehicle_id'], 'vehicle2')
        
        # Verify the annotator was called
        mock_annotator.annotate.assert_called_once_with(result, self.graph)


if __name__ == '__main__':
    unittest.main()