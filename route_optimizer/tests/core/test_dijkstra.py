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
            'F': {'A': 10.0} # Cycle back to A
        }
        
        # Graph with disconnected components
        self.disconnected_graph = {
            'A': {'B': 1.0},
            'B': {'A': 1.0},
            'C': {'D': 2.0},
            'D': {'C': 2.0}
        }

        # Empty graph
        self.empty_graph = {}

        # Graph with a single node
        self.single_node_graph = {
            'A': {}
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
        # Test A to F: A -> B -> C -> E -> F (1+2+3+1 = 7)
        path, distance = self.path_finder.calculate_shortest_path(
            self.complex_graph, 'A', 'F'
        )
        self.assertEqual(path, ['A', 'B', 'C', 'E', 'F'])
        self.assertEqual(distance, 7.0)

        # Test D to A: D -> E -> F -> A (2+1+10 = 13)
        path, distance = self.path_finder.calculate_shortest_path(
            self.complex_graph, 'D', 'A'
        )
        self.assertEqual(path, ['D', 'E', 'F', 'A'])
        self.assertEqual(distance, 13.0)

    def test_edge_cases_calculate_shortest_path(self):
        """Test edge cases for the calculate_shortest_path algorithm."""
        # Test path from a node to itself
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'A', 'A'
        )
        self.assertEqual(path, ['A'])
        self.assertEqual(distance, 0.0)
        
        # Test with non-existent start and end nodes
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

        # Test with end node exists but start doesn't
        path, distance = self.path_finder.calculate_shortest_path(
            self.simple_graph, 'X', 'A'
        )
        self.assertIsNone(path)
        self.assertIsNone(distance)

        # Test path in a disconnected graph (path does not exist between components)
        path, distance = self.path_finder.calculate_shortest_path(
            self.disconnected_graph, 'A', 'C'
        )
        self.assertIsNone(path)
        self.assertIsNone(distance)

        # Test path within a component of a disconnected graph
        path, distance = self.path_finder.calculate_shortest_path(
            self.disconnected_graph, 'C', 'D'
        )
        self.assertEqual(path, ['C', 'D'])
        self.assertEqual(distance, 2.0)

        # Test with empty graph
        path, distance = self.path_finder.calculate_shortest_path(
            self.empty_graph, 'A', 'B'
        )
        self.assertIsNone(path)
        self.assertIsNone(distance)

        # Test with single node graph (path to self)
        path, distance = self.path_finder.calculate_shortest_path(
            self.single_node_graph, 'A', 'A'
        )
        self.assertEqual(path, ['A'])
        self.assertEqual(distance, 0.0)

        # Test with single node graph (path to non-existent node)
        path, distance = self.path_finder.calculate_shortest_path(
            self.single_node_graph, 'A', 'B'
        )
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_all_shortest_paths_simple_graph(self):
        """Test calculating all shortest paths between nodes in the simple graph."""
        nodes = ['A', 'B', 'C', 'D', 'E'] 
        all_paths = self.path_finder.calculate_all_shortest_paths(self.simple_graph, nodes)

        # Structure checks
        for start_node in nodes:
            self.assertIn(start_node, all_paths)
            for end_node in nodes:
                self.assertIn(end_node, all_paths[start_node])
                self.assertIn('path', all_paths[start_node][end_node])
                self.assertIn('distance', all_paths[start_node][end_node])

        # Shortest path from A to C: A → B → C
        self.assertEqual(all_paths['A']['C']['path'], ['A', 'B', 'C'])
        self.assertEqual(all_paths['A']['C']['distance'], 3.0)

        # Shortest path from A to E: A → B → E (1 + 3 = 4.0)
        self.assertEqual(all_paths['A']['E']['path'], ['A', 'B', 'E'])
        self.assertEqual(all_paths['A']['E']['distance'], 4.0)

        # Shortest path from D to E: D → E
        self.assertEqual(all_paths['D']['E']['path'], ['D', 'E'])
        self.assertEqual(all_paths['D']['E']['distance'], 5.0)

        # Path from E to any other node is impossible (E has no outbound edges in simple_graph)
        for target_node in ['A', 'B', 'C', 'D']:
            self.assertIsNone(all_paths['E'][target_node]['path'])
            self.assertEqual(all_paths['E'][target_node]['distance'], float('inf'))

        # Self-paths
        for node in nodes:
            self.assertEqual(all_paths[node][node]['path'], [node])
            self.assertEqual(all_paths[node][node]['distance'], 0)

    def test_all_shortest_paths_complex_graph(self):
        """Test calculating all shortest paths in a more complex graph."""
        nodes = ['A', 'D', 'F']
        all_paths = self.path_finder.calculate_all_shortest_paths(self.complex_graph, nodes)

        # A to F: ['A', 'B', 'C', 'E', 'F'], 7.0
        self.assertEqual(all_paths['A']['F']['path'], ['A', 'B', 'C', 'E', 'F'])
        self.assertEqual(all_paths['A']['F']['distance'], 7.0)

        # D to A: ['D', 'E', 'F', 'A'], 13.0
        self.assertEqual(all_paths['D']['A']['path'], ['D', 'E', 'F', 'A'])
        self.assertEqual(all_paths['D']['A']['distance'], 13.0)
        
        # F to D: F -> A (10) -> B (1) -> C (2) -> D (1). Total: 14. Path: [F,A,B,C,D]
        self.assertEqual(all_paths['F']['D']['path'], ['F', 'A', 'B', 'C', 'D'])
        self.assertEqual(all_paths['F']['D']['distance'], 14.0)

    def test_all_shortest_paths_edge_cases(self):
        """Test edge cases for calculate_all_shortest_paths."""
        # Test with a node in 'nodes' list that is not in the graph dictionary
        nodes_with_unknown = ['A', 'X'] # X not in simple_graph
        all_paths = self.path_finder.calculate_all_shortest_paths(self.simple_graph, nodes_with_unknown)
        
        self.assertEqual(all_paths['A']['X']['path'], None)
        self.assertEqual(all_paths['A']['X']['distance'], float('inf'))
        self.assertEqual(all_paths['X']['A']['path'], None)
        self.assertEqual(all_paths['X']['A']['distance'], float('inf'))
        self.assertEqual(all_paths['X']['X']['path'], None) # because X not in graph
        self.assertEqual(all_paths['X']['X']['distance'], float('inf'))

        # Test with empty graph
        all_paths_empty_graph = self.path_finder.calculate_all_shortest_paths(self.empty_graph, ['A', 'B'])
        self.assertEqual(all_paths_empty_graph['A']['B']['path'], None)
        self.assertEqual(all_paths_empty_graph['A']['B']['distance'], float('inf'))
        self.assertEqual(all_paths_empty_graph['A']['A']['path'], None)
        self.assertEqual(all_paths_empty_graph['A']['A']['distance'], float('inf'))

        # Test with empty nodes list
        all_paths_empty_nodes = self.path_finder.calculate_all_shortest_paths(self.simple_graph, [])
        self.assertEqual(all_paths_empty_nodes, {})

        # Test with single node in nodes list
        all_paths_single_node = self.path_finder.calculate_all_shortest_paths(self.simple_graph, ['A'])
        self.assertEqual(all_paths_single_node['A']['A']['path'], ['A'])
        self.assertEqual(all_paths_single_node['A']['A']['distance'], 0.0)
        
        # Test with single node graph and single node in list
        all_paths_single_node_graph_list = self.path_finder.calculate_all_shortest_paths(self.single_node_graph, ['A'])
        self.assertEqual(all_paths_single_node_graph_list['A']['A']['path'], ['A'])
        self.assertEqual(all_paths_single_node_graph_list['A']['A']['distance'], 0.0)

    def test_negative_weights_error_calculate_shortest_path(self):
        """Test that calculate_shortest_path raises ValueError for negative weights."""
        graph_with_negative = {
            'A': {'B': 1.0, 'C': 4.0},
            'B': {'C': -2.0}  # Negative weight
        }

        with self.assertRaisesRegex(ValueError, "Negative weight detected from 'B' to 'C' with weight -2.0"):
            self.path_finder.calculate_shortest_path(graph_with_negative, 'A', 'C')
        
    def test_negative_weights_error_calculate_all_shortest_paths(self):
        """Test that calculate_all_shortest_paths raises ValueError for negative weights."""
        graph_with_negative = {
            'A': {'B': 2.0},
            'B': {'C': -3.0},  # Negative weight
            'C': {'A': 1.0}
        }
        nodes = ['A', 'B', 'C']

        with self.assertRaisesRegex(ValueError, "Negative weight detected from 'B' to 'C' with weight -3.0"):
            self.path_finder.calculate_all_shortest_paths(graph_with_negative, nodes)


if __name__ == '__main__':
    unittest.main()