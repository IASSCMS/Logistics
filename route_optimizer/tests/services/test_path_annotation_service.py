from django.test import TestCase
from unittest.mock import MagicMock, patch
import logging

from route_optimizer.services.path_annotation_service import PathAnnotator
from route_optimizer.core.types_1 import OptimizationResult, DetailedRoute, RouteSegment
from route_optimizer.models import Vehicle
import numpy as np

# Suppress logging for cleaner test output if necessary
# logging.disable(logging.CRITICAL)


class DummyPathFinder:
    def calculate_shortest_path(self, graph, from_node, to_node):
        # Simple path finder that returns direct path and fixed distance
        # Can be overridden in specific tests using mocks
        if from_node == "Error" and to_node == "Node": # Specific case for exception testing
            raise ConnectionError("Simulated network error")
        if from_node == "NoPath" and to_node == "Node": # Specific case for no path
            return None, 0 
            
        # Default successful path
        if graph.get(from_node) and graph[from_node].get(to_node):
            return [from_node, to_node], graph[from_node][to_node]
        return None, None # Default if not in graph for simplicity in dummy


class PathAnnotatorTest(TestCase):
    def setUp(self):
        self.graph = {'A': {'B': 5, 'D': 10}, 'B': {'C': 5}, 'C': {'A': 5}, 'Error': {}, 'NoPath': {}}
        self.path_finder = DummyPathFinder() # Use the enhanced DummyPathFinder
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
        
        # Check first route (detailed_routes is a list of dicts)
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
        """Test _add_summary_statistics method's interaction with RouteStatsService."""
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
        """Test that _add_summary_statistics correctly derives stops from segments if missing."""
        # Create a result with segments (using 'from'/'to' keys as expected by this helper)
        # but no 'stops' key.
        result = {
            'detailed_routes': [
                {
                    'vehicle_id': 'vehicle1',
                    'segments': [
                        {'from': 'A', 'to': 'B', 'path': ['A', 'B'], 'distance': 5},
                        {'from': 'B', 'to': 'C', 'path': ['B', 'C'], 'distance': 5}
                    ]
                }
            ]
        }
        
        # Call _add_summary_statistics which should add 'stops'
        with patch('route_optimizer.services.route_stats_service.RouteStatsService.add_statistics'): # Mock to avoid its internal logic
            self.annotator._add_summary_statistics(result, self.vehicles)
        
        # Verify that stops were added
        self.assertIn('stops', result['detailed_routes'][0])
        self.assertEqual(result['detailed_routes'][0]['stops'], ['A', 'B', 'C'])

    def test_annotate_with_matrix(self):
        """Test annotate method with a distance matrix input instead of a graph."""
        # Create a simple distance matrix
        distance_matrix = np.array([
            [0, 5, 10],
            [5, 0, 5],
            [10, 5, 0]
        ])
        location_ids = ['A', 'B', 'C']
        
        # Create the matrix input for the annotator
        matrix_input = {
            'matrix': distance_matrix,
            'location_ids': location_ids
        }
        
        # Dictionary-based result for annotation
        result = {
            'routes': [['A', 'B', 'C']],
            'assigned_vehicles': {'vehicle1': 0}
        }
        
        # Use patch to mock the distance matrix to graph conversion
        with patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.distance_matrix_to_graph') as mock_convert:
            # Set up mock to return our test graph
            mock_convert.return_value = self.graph # Simulate conversion to the graph defined in setUp
            
            # Annotate the result
            annotated = self.annotator.annotate(result, matrix_input)
            
            # Verify the conversion was called
            mock_convert.assert_called_once_with(distance_matrix, location_ids)
            
            # Check the results (segments should be based on self.graph via DummyPathFinder)
            self.assertIn('detailed_routes', annotated)
            self.assertEqual(len(annotated['detailed_routes']), 1)
            route1 = annotated['detailed_routes'][0]
            self.assertEqual(len(route1['segments']), 2)
            self.assertEqual(route1['segments'][0]['distance'], 5) # A to B from self.graph
            self.assertEqual(route1['segments'][1]['distance'], 5) # B to C from self.graph

    def test_annotate_path_calculation_issues(self):
        """Test annotate method when path_finder has issues."""
        # Case 1: Path finder returns (None, distance) for a segment
        # Case 2: Path finder raises an exception for another segment
        result = {
            'routes': [['A', 'D', 'NoPath', 'Node', 'Error', 'Node']], # A->D (ok), D->NoPath (ok), NoPath->Node (no path), Error->Node (exception)
            'assigned_vehicles': {'vehicle1': 0}
        }
        
        # Path A->D exists in self.graph, distance 10
        # Path D->NoPath: DummyPathFinder default (None,None)
        # Path NoPath->Node: DummyPathFinder returns (None, 0)
        # Path Error->Node: DummyPathFinder raises ConnectionError
        
        annotated_result = self.annotator.annotate(result, self.graph)
        
        self.assertIn('detailed_routes', annotated_result)
        detailed_route = annotated_result['detailed_routes'][0]
        segments = detailed_route['segments']

        # Expected number of segments:
        # A->D (success)
        # D->NoPath (path_finder returns None, so this segment is skipped by `if path:` condition)
        # NoPath->Node (path_finder returns None for path, so this segment is skipped by `if path:` condition)
        # Error->Node (exception, placeholder segment added)
        # Total should be 2 segments: A->D and the error placeholder for Error->Node.
        self.assertEqual(len(segments), 2) 

        # Check the successful segment A->D
        segment_ad = segments[0]
        self.assertEqual(segment_ad['from_location'], 'A')
        self.assertEqual(segment_ad['to_location'], 'D')
        self.assertEqual(segment_ad['distance'], 10)
        self.assertNotIn('error', segment_ad)

        # Check the placeholder segment for Error->Node
        segment_error = segments[1]
        self.assertEqual(segment_error['from_location'], 'Error')
        self.assertEqual(segment_error['to_location'], 'Node')
        self.assertEqual(segment_error['distance'], 0.0) # Placeholder distance
        self.assertEqual(segment_error['path'], ['Error', 'Node']) # Placeholder path
        self.assertIn('error', segment_error)
        self.assertEqual(segment_error['error'], "Simulated network error")

    def test_annotate_empty_routes(self):
        """Test annotate with empty routes list."""
        result_dict = {'routes': [], 'assigned_vehicles': {}}
        annotated_dict = self.annotator.annotate(result_dict, self.graph)
        self.assertEqual(len(annotated_dict['detailed_routes']), 0)

        result_dto = OptimizationResult(status='success', routes=[], detailed_routes=[])
        annotated_dto = self.annotator.annotate(result_dto, self.graph)
        self.assertEqual(len(annotated_dto.detailed_routes), 0)

    def test_annotate_route_with_single_stop(self):
        """Test annotate with a route that has only one stop (should produce no segments)."""
        result_dict = {'routes': [['A']], 'assigned_vehicles': {'vehicle1': 0}}
        annotated_dict = self.annotator.annotate(result_dict, self.graph)
        self.assertEqual(len(annotated_dict['detailed_routes'][0]['segments']), 0)

        result_dto = OptimizationResult(status='success', routes=[['A']], detailed_routes=[])
        result_dto.assigned_vehicles = {'vehicle1':0}
        annotated_dto = self.annotator.annotate(result_dto, self.graph)
        self.assertEqual(len(annotated_dto.detailed_routes[0]['segments']), 0)

