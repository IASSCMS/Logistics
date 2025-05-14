"""
Tests for the optimization service.

This module contains comprehensive tests for the OptimizationService class.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import dataclasses

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.core.types_1 import Location, OptimizationResult, DetailedRoute
from route_optimizer.models import Vehicle, Delivery
from route_optimizer.core.distance_matrix import DistanceMatrixBuilder # Import for direct testing
from route_optimizer.core.constants import MAX_SAFE_DISTANCE


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
        from route_optimizer.core.constants import MAX_SAFE_DISTANCE
        self.max_safe_distance = MAX_SAFE_DISTANCE

    # --- Basic Optimization Tests ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    @patch('route_optimizer.services.path_annotation_service.PathAnnotator.annotate') # Mock annotator
    @patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics') # Mock stats
    def test_optimize_routes_basic(self, mock_add_stats, mock_annotate, mock_get_depot, mock_create_matrix):
        """Test basic route optimization without traffic or time windows."""
        # Set up mocks
        # create_distance_matrix now returns (distance_matrix, time_matrix, location_ids)
        # For basic non-API, time_matrix can be None.
        mock_create_matrix.return_value = (self.distance_matrix, None, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        
        expected_solver_result = OptimizationResult(
            status='success',
            routes=[['depot', 'customer1', 'customer2', 'depot'], ['depot', 'customer3', 'depot']], # Use actual IDs
            total_distance=6.0,
            total_cost=0.0, # Will be updated by stats service
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[], # Will be populated by _add_detailed_paths
            statistics={}
        )
        self.mock_vrp_solver.solve.return_value = expected_solver_result
        
        # Call the service
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        
        # Verify the result
        self.assertEqual(result.status, 'success')
        # total_distance is preserved from solver, total_cost is calculated by RouteStatsService
        self.assertEqual(result.total_distance, 6.0) 
        self.assertEqual(len(result.routes), 2) # Check the simple routes list
        self.assertEqual(len(result.unassigned_deliveries), 0)
        
        # Verify the mocks were called correctly
        mock_create_matrix.assert_called_once_with(
            self.locations,
            use_haversine=True,
            distance_calculation="haversine",
            use_api=False, # Default behavior when use_api=None
            api_key=None   # Default behavior
        )
        self.mock_vrp_solver.solve.assert_called_once()
        mock_get_depot.assert_called_once()
        mock_annotate.assert_called_once() # PathAnnotator should be called for successful results
        mock_add_stats.assert_called_once()  # RouteStatsService should be called

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.add_traffic_factors') # Patching add_traffic_factors
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    @patch('route_optimizer.services.path_annotation_service.PathAnnotator.annotate')
    @patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics')
    def test_optimize_routes_with_traffic(self, mock_add_stats, mock_annotate, mock_get_depot, mock_add_traffic, mock_create_matrix):
        """Test route optimization with traffic data."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, None, self.location_ids)
        # add_traffic_factors will be called with the original matrix and traffic_data
        # It should return the modified matrix
        modified_distance_matrix_due_to_traffic = self.distance_matrix * 1.2 # Example modification
        mock_add_traffic.return_value = modified_distance_matrix_due_to_traffic
        
        mock_get_depot.return_value = self.locations[0]
        
        expected_solver_result = OptimizationResult(
            status='success',
            routes=[['depot', 'customer1', 'customer2', 'depot'], ['depot', 'customer3', 'depot']],
            total_distance=8.0,  # Increased due to traffic
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        self.mock_vrp_solver.solve.return_value = expected_solver_result
        
        traffic_data = {(0, 1): 1.5, (1, 2): 1.2} # Using indices as per service expectation now
        
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            consider_traffic=True,
            traffic_data=traffic_data
        )
        
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.total_distance, 8.0)
        
        mock_create_matrix.assert_called_once()
        mock_add_traffic.assert_called_once_with(ANY, traffic_data) # Check it's called with the matrix and traffic_data
        # The first argument to add_traffic_factors is the sanitized distance_matrix, so use ANY.
        self.mock_vrp_solver.solve.assert_called_once()
        mock_annotate.assert_called_once()
        mock_add_stats.assert_called_once()

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    @patch('route_optimizer.services.path_annotation_service.PathAnnotator.annotate')
    @patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics')
    def test_optimize_routes_with_time_windows(self, mock_add_stats, mock_annotate, mock_get_depot, mock_create_matrix):
        """Test route optimization with time windows."""
        mock_create_matrix.return_value = (self.distance_matrix, None, self.location_ids)
        mock_get_depot.return_value = self.locations[0]
        
        expected_solver_result = OptimizationResult(
            status='success',
            routes=[['depot', 'customer1', 'customer2', 'depot'], ['depot', 'customer3', 'depot']],
            total_distance=6.0, # Assuming time windows don't change distance in this mock
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        self.mock_vrp_solver.solve_with_time_windows.return_value = expected_solver_result
        
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            consider_time_windows=True
        )
        
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.total_distance, 6.0)
        
        self.mock_vrp_solver.solve_with_time_windows.assert_called_once()
        self.mock_vrp_solver.solve.assert_not_called()
        mock_annotate.assert_called_once()
        mock_add_stats.assert_called_once()

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    def test_validation_errors(self, mock_get_depot, mock_create_matrix):
        """Test validation errors are handled correctly."""
        # This test assumes _validate_inputs raises ValueError, which is caught by optimize_routes
        # and an OptimizationResult with status 'error' is returned.
        
        # No need to mock create_matrix or get_depot if validation fails before them.
        # However, to ensure the error comes from _validate_inputs, we can let them be called.
        mock_create_matrix.return_value = (self.distance_matrix, None, self.location_ids) # Correct 3-tuple
        mock_get_depot.return_value = self.locations[0]

        invalid_locations = [
            Location(id="invalid", name="Invalid", latitude=None, longitude=None, is_depot=False)
        ]
        
        result = self.service.optimize_routes(
            locations=invalid_locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        
        self.assertEqual(result.status, 'error')
        self.assertIn('error', result.statistics)
        # The specific error message from _validate_inputs
        self.assertIn(f"Location invalid is missing latitude or longitude coordinates", result.statistics['error'])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    def test_exception_handling_from_vrp_solver(self, mock_get_depot, mock_create_matrix):
        """Should handle exceptions from VRP solver gracefully."""
        # Ensure create_distance_matrix returns a valid 3-tuple to avoid unpack error
        mock_create_matrix.return_value = (self.distance_matrix, None, self.location_ids) 
        mock_get_depot.return_value = self.locations[0]
        
        self.mock_vrp_solver.solve.side_effect = Exception("VRP Solver exploded")
        # Also mock solve_with_time_windows if it could be called
        self.mock_vrp_solver.solve_with_time_windows.side_effect = Exception("VRP Solver (TW) exploded")
        
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        
        self.assertEqual(result.status, 'error')
        self.assertEqual(len(result.routes), 0)
        # All deliveries should be unassigned if optimization fails
        self.assertEqual(len(result.unassigned_deliveries), len(self.deliveries)) 
        self.assertIn('error', result.statistics)
        self.assertIn('Optimization failed: VRP Solver exploded', result.statistics['error'])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix', side_effect=ValueError("Matrix creation error"))
    def test_exception_handling_from_matrix_creation(self, mock_create_matrix):
        """Test exception handling from distance matrix creation."""
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        self.assertEqual(result.status, 'error')
        self.assertIn('error', result.statistics)
        self.assertIn('Optimization failed: Matrix creation error', result.statistics['error'])


    # --- Helper Method Tests (Now testing static/external methods) ---

    def test_sanitize_distance_matrix_static(self): # Renamed for clarity
        """Test sanitizing distance matrix using DistanceMatrixBuilder."""
        matrix = np.array([
            [0.0, 1.0, float('inf'), -5.0],
            [1.0, 0.0, float('nan'), 2.0],
            [float('inf'), float('nan'), 0.0, 5000.0],
            [-5.0, 2.0, 5000.0, 0.0]
        ])
        
        # Call the static method from DistanceMatrixBuilder
        result = DistanceMatrixBuilder._sanitize_distance_matrix(matrix)
        
        self.assertEqual(result[0, 2], self.max_safe_distance)
        self.assertEqual(result[2, 0], self.max_safe_distance)
        self.assertEqual(result[1, 2], self.max_safe_distance)
        self.assertEqual(result[2, 1], self.max_safe_distance)
        self.assertEqual(result[0, 3], 0.0)
        self.assertEqual(result[3, 0], 0.0)
        # Values <= MAX_SAFE_DISTANCE should remain unchanged by capping
        self.assertEqual(result[2, 3], 5000.0) 
        self.assertEqual(result[3, 2], 5000.0)

    def test_add_traffic_factors_static(self): # Renamed and adapted
        """Test applying traffic factors using DistanceMatrixBuilder.add_traffic_factors."""
        matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0]
        ], dtype=float) # Ensure float for multiplication
        
        traffic_data = {
            (0, 1): 1.5,    # Normal factor
            (1, 2): 10.0,   # Excessive factor (should be capped by _apply_traffic_safely inside add_traffic_factors)
            (2, 0): -1.0,   # Invalid factor (should be set to minimum 1.0 by _apply_traffic_safely)
            (5, 5): 2.0     # Out of bounds index (should be ignored by add_traffic_factors)
        }
        
        # Call the static method from DistanceMatrixBuilder
        # Note: add_traffic_factors makes a copy, applies factors, then sanitizes.
        result = DistanceMatrixBuilder.add_traffic_factors(np.copy(matrix), traffic_data) 
        
        self.assertAlmostEqual(result[0, 1], 1.5)  # 1.0 * 1.5
        
        # add_traffic_factors calls _apply_traffic_safely which caps factor at max_safe_factor (default 5.0)
        # So, 3.0 * 5.0 = 15.0
        self.assertAlmostEqual(result[1, 2], 3.0 * 5.0) 
        
        # _apply_traffic_safely sets factor < 1.0 to 1.0, so 2.0 * 1.0 = 2.0
        self.assertAlmostEqual(result[2, 0], 2.0) 
        
        # Check out of bounds index was ignored
        self.assertAlmostEqual(result[0, 0], 0.0)

    def test_convert_to_optimization_result_static(self): # Renamed
        """Test converting dictionary to OptimizationResult using OptimizationResult.from_dict."""
        result_dict = {
            'status': 'success',
            'routes': [['loc1', 'loc2'], ['loc1', 'loc3']], # Using strings for location_ids
            'total_distance': 5.0,
            'total_cost': 150.0,
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1},
            'unassigned_deliveries': ['delivery3'],
            'detailed_routes': [], # Expect list of dicts here
            'statistics': {'total_time': 120}
        }
        
        result = OptimizationResult.from_dict(result_dict)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.routes, [['loc1', 'loc2'], ['loc1', 'loc3']])
        self.assertEqual(result.total_distance, 5.0)
        # ... other assertions
        self.assertEqual(result.detailed_routes, []) # from_dict will use default if key missing or type wrong

    def test_convert_empty_result_static(self): # Renamed
        """Test converting an empty or invalid result dictionary using OptimizationResult.from_dict."""
        result_empty = OptimizationResult.from_dict({})
        self.assertIsInstance(result_empty, OptimizationResult)
        self.assertEqual(result_empty.status, 'unknown') # Default status for empty dict
        
        result_none = OptimizationResult.from_dict(None)
        self.assertIsInstance(result_none, OptimizationResult)
        self.assertEqual(result_none.status, 'error')
        self.assertIn('Input data for OptimizationResult was None', result_none.statistics['error'])

    # --- External API Tests ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.depot_service.DepotService.get_nearest_depot')
    @patch('route_optimizer.services.traffic_service.TrafficService.create_road_graph') # For detailed paths
    @patch('route_optimizer.services.path_annotation_service.PathAnnotator.annotate')
    @patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics')
    def test_optimize_routes_with_api(self, mock_add_stats, mock_annotate, mock_create_graph, mock_get_depot, mock_create_matrix_builder):
        """Test optimization using external API."""
        mock_create_matrix_builder.return_value = (self.distance_matrix, np.array([]), self.location_ids) # API might return time matrix
        mock_get_depot.return_value = self.locations[0]
        mock_create_graph.return_value = self.graph # Mock the graph creation from TrafficService
        
        expected_solver_result = OptimizationResult(
            status='success',
            routes=[['depot', 'customer1', 'customer2', 'depot'], ['depot', 'customer3', 'depot']],
            total_distance=6.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        self.mock_vrp_solver.solve.return_value = expected_solver_result
        
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            use_api=True,
            api_key='test_api_key'
        )
        
        self.assertEqual(result.status, 'success')
        
        mock_create_matrix_builder.assert_called_once_with(
            self.locations, 
            use_api=True,         # This is passed correctly to create_distance_matrix
            api_key='test_api_key'
        )
        mock_create_graph.assert_called_once() # For detailed path generation
        mock_annotate.assert_called_once()
        mock_add_stats.assert_called_once()

    # --- Add Detailed Paths Tests ---

    def test_add_detailed_paths_optimization_result(self):
        """Test adding detailed paths to OptimizationResult."""
        # Create a sample optimization result
        initial_detailed_routes = [ # Example with one pre-filled detailed route for testing merge/append
            dataclasses.asdict(DetailedRoute(vehicle_id='vehicle_pre', stops=['A','B'], segments=[]))
        ]
        result_dto = OptimizationResult(
            status='success',
            routes=[['depot', 'customer1', 'depot'], ['depot', 'customer2', 'depot']], # Using actual IDs
            total_distance=4.0,
            total_cost=100.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1}, # vehicle1 maps to first route, vehicle2 to second
            unassigned_deliveries=[],
            detailed_routes=initial_detailed_routes, # Start with some existing detailed_routes
            statistics={}
        )
        
        mock_annotator_instance = MagicMock()
        # PathAnnotator.annotate is called with the result DTO and the graph
        # It modifies result.detailed_routes in place or reassigns it.
        
        # Call _add_detailed_paths
        # The _add_detailed_paths method populates detailed_routes from routes if empty,
        # then calls the annotator.
        with patch('route_optimizer.services.optimization_service.PathAnnotator') as MockPathAnnotator:
            MockPathAnnotator.return_value = mock_annotator_instance # Mock the instance
            
            # _add_detailed_paths internally creates DetailedRoute objects and converts to dicts for the list
            # if result.routes is present and result.detailed_routes is empty.
            # Let's simulate that detailed_routes is initially empty to test that path.
            result_dto.detailed_routes = []
            
            enriched_result = self.service._add_detailed_paths(
                result_dto, # Pass the DTO directly
                self.graph,
                self.location_ids # Pass location_ids for stop conversion
                # annotator is created internally if None
            )

        self.assertTrue(isinstance(enriched_result, OptimizationResult)) # Should still be a DTO
        self.assertTrue(hasattr(enriched_result, 'detailed_routes'))
        
        # After _add_detailed_paths, detailed_routes should be populated based on result.routes
        # It should have 2 entries from the 'routes' list.
        self.assertEqual(len(enriched_result.detailed_routes), 2) 
        
        # Check vehicle_id assignment in the created detailed_route dicts
        # detailed_routes now contains list of DICTs, not DTOs, after _add_detailed_paths logic
        self.assertEqual(enriched_result.detailed_routes[0]['vehicle_id'], 'vehicle1')
        self.assertEqual(enriched_result.detailed_routes[0]['stops'], ['depot', 'customer1', 'depot'])
        self.assertEqual(enriched_result.detailed_routes[1]['vehicle_id'], 'vehicle2')
        self.assertEqual(enriched_result.detailed_routes[1]['stops'], ['depot', 'customer2', 'depot'])
        
        # Verify the mock annotator instance's annotate method was called
        mock_annotator_instance.annotate.assert_called_once_with(result_dto, self.graph)


    def test_add_detailed_paths_dict(self):
        """Test adding detailed paths to result dictionary."""
        result_dict = {
            'status': 'success',
            'routes': [['depot', 'customer1', 'depot'], ['depot', 'customer2', 'depot']],
            'total_distance': 4.0,
            'total_cost': 100.0,
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1},
            'unassigned_deliveries': [],
            'detailed_routes': [], # Start empty
            'statistics': {}
        }
        
        mock_annotator_instance = MagicMock()
        with patch('route_optimizer.services.optimization_service.PathAnnotator') as MockPathAnnotator:
            MockPathAnnotator.return_value = mock_annotator_instance
            
            enriched_result = self.service._add_detailed_paths(
                result_dict,
                self.graph,
                self.location_ids
            )

        self.assertIn('detailed_routes', enriched_result)
        self.assertEqual(len(enriched_result['detailed_routes']), 2)
        self.assertEqual(enriched_result['detailed_routes'][0]['vehicle_id'], 'vehicle1')
        self.assertEqual(enriched_result['detailed_routes'][1]['vehicle_id'], 'vehicle2')
        
        mock_annotator_instance.annotate.assert_called_once_with(result_dict, self.graph)

if __name__ == '__main__':
    unittest.main()