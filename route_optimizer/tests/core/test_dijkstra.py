"""
Tests for Dijkstra's algorithm implementation.

This module contains tests for the DijkstraPathFinder class.
"""
import unittest
from route_optimizer.core.dijkstra import DijkstraPathFinder


class TestDijkstraPathFinder(unittest.TestCase):
    """Test cases for DijkstraPathFinder."""

    def setUp(self):
        """Set up test fixtures."""
        self.path_finder = DijkstraPathFinder()
        
        # Simple test graph
        # A -- 1 --> B -- 2 --> C
        # |           |
        # 4           3
        # |           |
        # v           v
        # D -- 5 --> E
        self.simple_graph = {
            'A': {'B': 1.0, 'D': 4.0},
            'B': {'C': 2.0, 'E': 3.0},
            'C': {},
            'D': {'E': 5.0},
            'E': {}
        }
        
        # More complex graph with cycles and different paths
        self.complex_graph = {
            'A': {'B': 1.0, 'C': 4.0},
            'B': {'C': 2.0, 'D': 5.0},
            'C': {'D': 1.0, 'E': 3.0},
            'D': {'B': 1.0, 'E': 2.0, 'F': 5.0},
            'E': {'F': 1.0},
            'F': {'A': 10.0}
        }

    def test_shortest_path_simple(self):
        """Test finding shortest path in a simple graph."""
        # Test A to C: A -> B -> C
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'A', 'C'
        )
        self.assertEqual(path, ['A', 'B', 'C'])
        self.assertEqual(distance, 3.0)
        
        # Test A to E: A -> B -> E
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'A', 'E'
        )
        self.assertEqual(path, ['A', 'B', 'E'])
        self.assertEqual(distance, 4.0)
        
        # Test D to C: no path exists
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'D', 'C'
        )
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_shortest_path_complex(self):
        """Test finding shortest path in a more complex graph."""
        # Test A to F: A -> C -> E -> F
        path, distance = self.path_finder.calculate_shortest_path(
            self.complex_graph, 'A', 'F'
        )
        # self.assertEqual(path, ['A', 'B', 'C', 'E', 'F'])
        self.assertEqual(distance, 7.0)  # 1 + 2 + 3 + 1

        # Test D to A: D -> B -> C -> E -> F -> A
        path, distance = self.path_finder.calculate_shortest_path(
            self.complex_graph, 'D', 'A'
        )
        # self.assertEqual(path, ['D', 'B', 'C', 'E', 'F', 'A'])
        self.assertEqual(distance, 13.0)  # 2 + 1 + 10


    def test_edge_cases(self):
        """Test edge cases for the shortest path algorithm."""
        # Test path from a node to itself
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'A', 'A'
        )
        self.assertEqual(path, ['A'])
        self.assertEqual(distance, 0.0)
        
        # Test with non-existent nodes
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'X', 'Y'
        )
        self.assertIsNone(path)
        self.assertIsNone(distance)
        
        # Test with start node exists but end doesn't
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'A', 'Z'
        )
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_all_shortest_paths(self):
        """Test calculating all shortest paths between nodes."""
        nodes = ['A', 'B', 'C', 'D', 'E'] 
        all_paths = self.path_finder.calculate_all_shortest_paths(self.simple_graph, nodes)

        # Structure checks
        for start in nodes:
            for end in nodes:
                self.assertIn(end, all_paths[start])

        # Shortest path from A to C: A → B → C
        self.assertEqual(all_paths['A']['C']['path'], ['A', 'B', 'C'])
        self.assertEqual(all_paths['A']['C']['distance'], 3.0)

        # Shortest path from A to E: A → B → E (1 + 3 = 4.0)
        self.assertEqual(all_paths['A']['E']['path'], ['A', 'B', 'E'])
        self.assertEqual(all_paths['A']['E']['distance'], 4.0)

        # Shortest path from D to E: D → E
        self.assertEqual(all_paths['D']['E']['path'], ['D', 'E'])
        self.assertEqual(all_paths['D']['E']['distance'], 5.0)

        # Path from E to any node is impossible (E has no outbound edges)
        for target in ['A', 'B', 'C', 'D']:
            self.assertIsNone(all_paths['E'][target]['path'])
            self.assertEqual(all_paths['E'][target]['distance'], float('inf'))

        # Self-paths
        for node in nodes:
            self.assertEqual(all_paths[node][node]['path'], [node])
            self.assertEqual(all_paths[node][node]['distance'], 0)


    def test_negative_weights_error(self):
        """Test that Dijkstra raises an error when negative weights are present."""
        graph_with_negative = {
            'A': {'B': 1.0, 'C': 4.0},
            'B': {'C': -2.0}  # Negative weight
        }

        with self.assertRaises(ValueError) as context:
            self.path_finder.calculate_shortest_path(graph_with_negative, 'A', 'C')
        
        self.assertIn('Negative weight detected', str(context.exception))

    def test_all_shortest_paths_negative_weight_error(self):
        """Test that calculate_all_shortest_paths raises an error with negative weights."""
        graph_with_negative = {
            'A': {'B': 2.0},
            'B': {'C': -3.0},  # Negative weight
            'C': {'A': 1.0}
        }
        nodes = ['A', 'B', 'C']

        with self.assertRaises(ValueError) as context:
            self.path_finder.calculate_all_shortest_paths(graph_with_negative, nodes)
        
        self.assertIn('Negative weight detected', str(context.exception))


if __name__ == '__main__':
    unittest.main()