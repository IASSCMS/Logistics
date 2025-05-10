"""
Tests for the optimization service.

This module contains comprehensive tests for the OptimizationService class.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.core.types_1 import Location, OptimizationResult
from route_optimizer.models import Vehicle, Delivery


class TestOptimizationService(unittest.TestCase):
    """Test cases for OptimizationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = OptimizationService()
        
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
            "depot": {"customer1": 1.0, "customer2": 1.0, "customer3": 1.4},
            "customer1": {"depot": 1.0, "customer2": 1.4, "customer3": 1.0},
            "customer2": {"depot": 1.0, "customer1": 1.4, "customer3": 1.0},
            "customer3": {"depot": 1.4, "customer1": 1.0, "customer2": 1.0}
        }

    # --- Basic Optimization Tests ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_routes_basic(self, mock_solve, mock_create_matrix):
        """Test basic route optimization without traffic or time windows."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_solve.return_value = OptimizationResult(
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
        mock_solve.assert_called_once()

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_routes_with_traffic(self, mock_solve, mock_create_matrix):
        """Test route optimization with traffic data."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_solve.return_value = OptimizationResult(
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
        
        # Call the service
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
        
        # Verify the mock was called correctly
        mock_create_matrix.assert_called_once()
        mock_solve.assert_called_once()

    # --- Edge Case Tests ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_with_no_deliveries(self, mock_solve, mock_create_matrix):
        """Should handle when there are no deliveries."""
        mock_create_matrix.return_value = (np.array([[0.0]]), ["depot"])
        mock_solve.return_value = OptimizationResult(
            status='success',
            routes=[[0]],
            total_distance=0.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        result = self.service.optimize_routes(
            locations=[self.locations[0]],  # Only depot
            vehicles=self.vehicles[:1],  # First vehicle
            deliveries=[]
        )
        
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.total_distance, 0.0)
        self.assertEqual(result.routes[0], [0])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_invalid_depot_index(self, mock_solve, mock_create_matrix):
        """Should fall back to index 0 when no depot is marked."""
        locations = [Location(id="node0", name="Node 0", latitude=0.0, longitude=0.0)]
        mock_create_matrix.return_value = (np.array([[0.0]]), ["node0"])
        mock_solve.return_value = OptimizationResult(
            status='success',
            routes=[[0]],
            total_distance=0.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        result = self.service.optimize_routes(
            locations=locations,
            vehicles=self.vehicles[:1],  # First vehicle
            deliveries=[]
        )
        
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.routes[0], [0])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_failure_case(self, mock_solve, mock_create_matrix):
        """Should return failure result if solver fails."""
        mock_create_matrix.return_value = (np.array([[0.0]]), ["depot"])
        mock_solve.return_value = OptimizationResult(
            status='failed',
            routes=[],
            total_distance=0.0,
            total_cost=0.0,
            assigned_vehicles={},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={'error': 'No solution found!'}
        )
        
        result = self.service.optimize_routes(
            locations=[self.locations[0]],  # Only depot
            vehicles=self.vehicles[:1],  # First vehicle
            deliveries=[]
        )
        
        self.assertEqual(result.status, 'failed')
        self.assertEqual(len(result.routes), 0)
        self.assertIn('error', result.statistics)

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_exception_handling(self, mock_solve, mock_create_matrix):
        """Should handle exceptions gracefully."""
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_solve.side_effect = Exception("Test exception")
        
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

    # --- Enrichment Tests ---

    def test_add_detailed_paths_basic(self):
        """Test adding detailed paths to optimization result."""
        # Create a basic optimization result
        result = OptimizationResult(
            status='success',
            routes=[[0, 1, 2, 0]],
            total_distance=0.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        # Mock path finder
        mock_pathfinder = MagicMock()
        mock_pathfinder.calculate_shortest_path.side_effect = lambda graph, from_node, to_node: (
            # Return hardcoded paths for testing
            ["depot", "customer1"] if from_node == "depot" and to_node == "customer1" else
            ["customer1", "customer2"] if from_node == "customer1" and to_node == "customer2" else
            ["customer2", "depot"] if from_node == "customer2" and to_node == "depot" else
            None  # Default case
        )
        
        # Patch pathfinder
        with patch.object(self.service, 'path_finder', mock_pathfinder):
            # Call the method to add detailed paths
            self.service._add_detailed_paths(
                result, 
                self.graph, 
                self.location_ids
            )
        
        # Verify detailed routes were added
        self.assertTrue(result.detailed_routes)
        self.assertEqual(len(result.detailed_routes), 1)  # One route
        
        # Check vehicle assignment
        self.assertEqual(result.detailed_routes[0]['vehicle_id'], 'vehicle1')
        
        # Check segments
        segments = result.detailed_routes[0]['segments']
        self.assertEqual(len(segments), 3)  # Three segments in the route
        
        # Check specific segment details
        self.assertEqual(segments[0]['from'], 'depot')
        self.assertEqual(segments[0]['to'], 'customer1')
        self.assertEqual(segments[0]['path'], ['depot', 'customer1'])
        
        self.assertEqual(segments[1]['from'], 'customer1')
        self.assertEqual(segments[1]['to'], 'customer2')
        self.assertEqual(segments[1]['path'], ['customer1', 'customer2'])
        
        self.assertEqual(segments[2]['from'], 'customer2')
        self.assertEqual(segments[2]['to'], 'depot')
        self.assertEqual(segments[2]['path'], ['customer2', 'depot'])

    @patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics')
    def test_add_summary_statistics(self, mock_add_stats):
        """Test adding summary statistics to optimization result."""
        # Create a basic result
        result = OptimizationResult(
            status='success',
            routes=[[0, 1, 2, 0]],
            total_distance=3.4,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0},
            unassigned_deliveries=[],
            detailed_routes=[{
                'vehicle_id': 'vehicle1',
                'stops': ['depot', 'customer1', 'customer2', 'depot'],
                'segments': [
                    {'distance': 1.0},
                    {'distance': 1.4},
                    {'distance': 1.0}
                ]
            }],
            statistics={}
        )
        
        # Mock the add_statistics method to do nothing
        mock_add_stats.side_effect = lambda r, v: None
        
        # Call the method
        self.service._add_summary_statistics(result, self.vehicles)
        
        # Verify the mock was called with correct arguments
        mock_add_stats.assert_called_once()
        args = mock_add_stats.call_args[0]
        self.assertEqual(args[0], result)
        self.assertEqual(args[1], self.vehicles)

    # --- Integration Test ---

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    @patch('route_optimizer.services.path_annotation_service.PathAnnotator.annotate')
    @patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics')
    def test_optimize_routes_end_to_end(self, mock_add_stats, mock_annotate, mock_solve, mock_create_matrix):
        """Test the full route optimization flow."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_solve.return_value = OptimizationResult(
            status='success',
            routes=[[0, 1, 2, 0], [0, 3, 0]],
            total_distance=6.0,
            total_cost=0.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        mock_annotate.side_effect = lambda r, g: None
        mock_add_stats.side_effect = lambda r, v: None
        
        # Call the service
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            use_api=False
        )
        
        # Verify the result
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.total_distance, 6.0)
        
        # Verify all mocks were called
        mock_create_matrix.assert_called_once()
        mock_solve.assert_called_once()
        mock_annotate.assert_called_once()


if __name__ == '__main__':
    unittest.main()