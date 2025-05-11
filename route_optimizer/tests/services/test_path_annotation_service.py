from django.test import TestCase
from unittest.mock import MagicMock, patch

from route_optimizer.services.path_annotation_service import PathAnnotator
from route_optimizer.core.types_1 import OptimizationResult, DetailedRoute, RouteSegment
from route_optimizer.models import Vehicle
import numpy as np


class DummyPathFinder:
    def calculate_shortest_path(self, graph, from_node, to_node):
        # Simple path finder that returns direct path and fixed distance
        return [from_node, to_node], 5


class PathAnnotatorTest(TestCase):
    def setUp(self):
        self.graph = {'A': {'B': 5}, 'B': {'C': 5}, 'C': {'A': 5}}
        self.path_finder = DummyPathFinder()
        self.annotator = PathAnnotator(self.path_finder)
        
        # Create test vehicles
        self.vehicles = [
            Vehicle(id="vehicle1", capacity=100, fixed_cost=10, cost_per_km=0.5, 
                   start_location_id="A", end_location_id="A"),
            Vehicle(id="vehicle2", capacity=50, fixed_cost=5, cost_per_km=0.3,
                   start_location_id="B", end_location_id="B")
        ]

    def test_annotate_with_dict(self):
        """Test annotate method with dictionary result"""
        # Dictionary-based result
        result = {
            'routes': [['A', 'B', 'C'], ['B', 'C']],
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1}
        }
        
        # Annotate the result
        annotated = self.annotator.annotate(result, self.graph)
        
        # Verify result structure
        self.assertIn('detailed_routes', annotated)
        self.assertEqual(len(annotated['detailed_routes']), 2)
        
        # Check first route
        route1 = annotated['detailed_routes'][0]
        self.assertEqual(route1['vehicle_id'], 'vehicle1')
        self.assertEqual(route1['stops'], ['A', 'B', 'C'])
        self.assertEqual(len(route1['segments']), 2)
        
        # Update field name to match implementation
        self.assertEqual(route1['segments'][0]['from_location'], 'A')
        self.assertEqual(route1['segments'][0]['to_location'], 'B')
        self.assertEqual(route1['segments'][0]['distance'], 5)
        
        # Check second route
        route2 = annotated['detailed_routes'][1]
        self.assertEqual(route2['vehicle_id'], 'vehicle2')
        self.assertEqual(route2['stops'], ['B', 'C'])
        self.assertEqual(len(route2['segments']), 1)

    def test_annotate_with_optimization_result(self):
        """Test annotate method with OptimizationResult object"""
        # Create OptimizationResult
        result = OptimizationResult(
            status='success',
            routes=[['A', 'B', 'C'], ['B', 'C']],
            total_distance=15.0,
            total_cost=20.0,
            assigned_vehicles={'vehicle1': 0, 'vehicle2': 1},
            unassigned_deliveries=[],
            detailed_routes=[],
            statistics={}
        )
        
        # Annotate the result
        annotated = self.annotator.annotate(result, self.graph)
        
        # Verify result structure
        self.assertTrue(hasattr(annotated, 'detailed_routes'))
        self.assertEqual(len(annotated.detailed_routes), 2)
        
        # Check first route
        route1 = annotated.detailed_routes[0]
        self.assertEqual(route1['vehicle_id'], 'vehicle1')
        self.assertEqual(route1['stops'], ['A', 'B', 'C'])
        self.assertEqual(len(route1['segments']), 2)
        
        # Check second route
        route2 = annotated.detailed_routes[1]
        self.assertEqual(route2['vehicle_id'], 'vehicle2')
        self.assertEqual(route2['stops'], ['B', 'C'])
        self.assertEqual(len(route2['segments']), 1)

    def test_add_summary_statistics(self):
        """Test add_summary_statistics method"""
        # Dictionary-based result with detailed routes
        result = {
            'detailed_routes': [
                {'vehicle_id': 'vehicle1', 'stops': ['A', 'B', 'C'], 'segments': []}
            ],
            'assigned_vehicles': {'vehicle1': 0}
        }
        
        # Use patch to mock the RouteStatsService.add_statistics method
        with patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics') as mock_add_stats:
            # Call the method
            self.annotator._add_summary_statistics(result, self.vehicles)
            
            # Check that the statistics service was called
            mock_add_stats.assert_called_once_with(result, self.vehicles)

    def test_handle_missing_stops(self):
        """Test that missing stops are handled correctly"""
        # Create a result with segments but no stops
        result = {
            'detailed_routes': [
                {
                    'vehicle_id': 'vehicle1',
                    'segments': [
                        # Update field names to match implementation
                        {'from': 'A', 'to': 'B', 'path': ['A', 'B'], 'distance': 5},
                        {'from': 'B', 'to': 'C', 'path': ['B', 'C'], 'distance': 5}
                    ]
                }
            ]
        }
        
        # Call add_summary_statistics which should add stops
        with patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics') as mock_add_stats:
            self.annotator._add_summary_statistics(result, self.vehicles)
        
        # Verify that stops were added
        self.assertIn('stops', result['detailed_routes'][0])
        self.assertEqual(result['detailed_routes'][0]['stops'], ['A', 'B', 'C'])

    def test_annotate_with_matrix(self):
        """Test annotate method with a distance matrix instead of a graph"""
        # Create a simple distance matrix
        distance_matrix = np.array([
            [0, 5, 10],
            [5, 0, 5],
            [10, 5, 0]
        ])
        location_ids = ['A', 'B', 'C']
        
        # Create the matrix input
        matrix_input = {
            'matrix': distance_matrix,
            'location_ids': location_ids
        }
        
        # Dictionary-based result
        result = {
            'routes': [['A', 'B', 'C']],
            'assigned_vehicles': {'vehicle1': 0}
        }
        
        # Use patch to mock the distance matrix to graph conversion
        with patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.distance_matrix_to_graph') as mock_convert:
            # Set up mock to return our test graph
            mock_convert.return_value = self.graph
            
            # Annotate the result
            annotated = self.annotator.annotate(result, matrix_input)
            
            # Verify the conversion was called
            mock_convert.assert_called_once_with(distance_matrix, location_ids)
            
            # Check the results
            self.assertIn('detailed_routes', annotated)
            self.assertEqual(len(annotated['detailed_routes']), 1)
            self.assertEqual(len(annotated['detailed_routes'][0]['segments']), 2)
