import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.core.ortools_optimizer import Vehicle, Delivery
from route_optimizer.core.distance_matrix import Location
from route_optimizer.core.dijkstra import DijkstraPathFinder

class TestOptimizationServiceEnrichment(unittest.TestCase):
    """Test cases for OptimizationService enrichment methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = OptimizationService()
        
        # Sample locations
        self.locations = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True),
            Location(id="customer1", name="Customer 1", latitude=1.0, longitude=0.0),
            Location(id="customer2", name="Customer 2", latitude=0.0, longitude=1.0)
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
            )
        ]
        
        # Mock distance matrix and graph
        self.distance_matrix = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.4],
            [1.0, 1.4, 0.0]
        ])
        self.location_ids = ["depot", "customer1", "customer2"]
        
        # Sample graph for pathfinding
        self.graph = {
            "depot": {"customer1": 1.0, "customer2": 1.0},
            "customer1": {"depot": 1.0, "customer2": 1.4},
            "customer2": {"depot": 1.0, "customer1": 1.4}
        }

    def test_add_detailed_paths_basic(self):
        """Test adding detailed paths to optimization result."""
        # Create a basic optimization result
        result = {
            'status': 'success',
            'routes': [[0, 1, 2, 0]],
            'assigned_vehicles': {'vehicle1': 0}
        }
        
        # Create a mock pathfinder
        mock_pathfinder = MagicMock()
        mock_pathfinder.calculate_shortest_path.side_effect = lambda graph, from_node, to_node: (
            # Return hardcoded paths for testing
            ["depot", "customer1"] if from_node == "depot" and to_node == "customer1" else
            ["customer1", "customer2"] if from_node == "customer1" and to_node == "customer2" else
            ["customer2", "depot"] if from_node == "customer2" and to_node == "depot" else
            None  # Default case
        )
        
        # Call the method to add detailed paths
        with patch.object(self.service, '_create_pathfinder', return_value=mock_pathfinder):
            detailed_result = self.service._add_detailed_paths(
                result, 
                self.graph, 
                self.location_ids
            )
        
        # Verify detailed routes were added
        self.assertIn('detailed_routes', detailed_result)
        self.assertEqual(len(detailed_result['detailed_routes']), 1)  # One route
        
        # Check vehicle assignment
        self.assertEqual(detailed_result['detailed_routes'][0]['vehicle_id'], 'vehicle1')
        
        # Check segments
        segments = detailed_result['detailed_routes'][0]['segments']
        self.assertEqual(len(segments), 3)  # Three segments in the route
        
        # Check specific segment details
        self.assertEqual(segments[0]['from'], 'depot')
        self.assertEqual(segments[0]['to'], 'customer1')
        self.assertEqual(segments[0]['path'], ['depot', 'customer1'])
        self.assertEqual(segments[0]['distance'], 1.0)
        
        self.assertEqual(segments[1]['from'], 'customer1')
        self.assertEqual(segments[1]['to'], 'customer2')
        self.assertEqual(segments[1]['path'], ['customer1', 'customer2'])
        self.assertEqual(segments[1]['distance'], 1.4)

    def test_add_summary_statistics(self):
        """Test adding summary statistics to optimization result."""
        # Create an optimization result with detailed routes
        result = {
            'status': 'success',
            'assigned_vehicles': {'vehicle1': 0},
            'detailed_routes': [
                {
                    'vehicle_id': 'vehicle1',
                    'segments': [
                        {'distance': 1.0},
                        {'distance': 1.4},
                        {'distance': 1.0}
                    ]
                }
            ]
        }
        
        # Call the method to add summary statistics
        detailed_result = self.service._add_summary_statistics(result, self.vehicles)
        
        # Verify summary statistics were added
        self.assertIn('vehicle_costs', detailed_result)
        self.assertIn('total_cost', detailed_result)
        
        # Check vehicle costs
        self.assertIn('vehicle1', detailed_result['vehicle_costs'])
        vehicle_cost = detailed_result['vehicle_costs']['vehicle1']
        
        # Check distance calculation (1.0 + 1.4 + 1.0 = 3.4)
        self.assertAlmostEqual(vehicle_cost['distance'], 3.4)
        
        # Check cost calculation (fixed_cost + distance * cost_per_km = 100 + 3.4 * 2 = 106.8)
        self.assertAlmostEqual(vehicle_cost['cost'], 106.8)
        
        # Check total cost
        self.assertAlmostEqual(detailed_result['total_cost'], 106.8)

    def test_add_summary_statistics_no_detailed_routes(self):
        """Test adding summary statistics when detailed routes are missing."""
        # Create an optimization result without detailed routes
        result = {
            'status': 'success',
            'assigned_vehicles': {'vehicle1': 0}
        }
        
        # Call the method to add summary statistics
        detailed_result = self.service._add_summary_statistics(result, self.vehicles)
        
        # Verify that the method handles the missing data gracefully
        self.assertIn('vehicle_costs', detailed_result)
        self.assertIn('total_cost', detailed_result)
        self.assertEqual(detailed_result['total_cost'], 0)

    def test_add_summary_statistics_vehicle_not_found(self):
        """Test adding summary statistics when a vehicle is not found."""
        # Create an optimization result with a non-existent vehicle
        result = {
            'status': 'success',
            'assigned_vehicles': {'non_existent_vehicle': 0},
            'detailed_routes': [
                {
                    'vehicle_id': 'non_existent_vehicle',
                    'segments': [
                        {'distance': 1.0}
                    ]
                }
            ]
        }
        
        # Call the method to add summary statistics
        detailed_result = self.service._add_summary_statistics(result, self.vehicles)
        
        # Verify that the method handles the missing vehicle gracefully
        self.assertIn('vehicle_costs', detailed_result)
        self.assertIn('total_cost', detailed_result)
        self.assertEqual(detailed_result['total_cost'], 0)

if __name__ == '__main__':
    unittest.main()
