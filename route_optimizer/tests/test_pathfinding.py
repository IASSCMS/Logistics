"""
Tests for Dijkstra's/Bellman-Ford/Pathfinder algorithm implementation.
"""
import unittest
import pytest
from route_optimizer.core.dijkstra import DijkstraPathFinder
from route_optimizer.core.shortest_path import find_shortest_path
from route_optimizer.core.bellman_ford import BellmanFordPathFinder

class TestDijkstraPathFinder(unittest.TestCase):
    """Test cases for DijkstraPathFinder."""

    def setUp(self):
        # Graph is represented as an adjacency list: {node: [(neighbor, weight), ...]}
        self.graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('C', 2), ('D', 5)],
            'C': [('D', 1)],
            'D': []
        }
        self.pathfinder = DijkstraPathFinder()

    def test_shortest_path(self):
        """Test finding shortest path in a graph."""
        path, cost = self.pathfinder.calculate_shortest_path(self.graph, 'A', 'D')
        self.assertEqual(path, ['A', 'B', 'C', 'D'])
        self.assertEqual(cost, 4)

    def test_edge_cases(self):
        """Test edge cases for the shortest path algorithm."""
        # Test path from a node to itself
        path, cost = self.path_finder.calculate_shortest_path(self.graph, 'A', 'A')
        self.assertEqual(path, ['A'])
        self.assertEqual(cost, 0.0)
        
        # Test with non-existent nodes
        with self.assertRaises(ValueError):
            self.pathfinder.calculate_shortest_path(self.graph, 'A', 'Z')

        with self.assertRaises(ValueError):
            self.pathfinder.calculate_shortest_path(self.graph, 'Z', 'A')

        # Test with unreachable node
        # Add a disconnected node 'E'
        self.graph['E'] = []
        self.pathfinder = DijkstraPathFinder(self.graph)  # Rebuild with updated graph
        with self.assertRaises(ValueError):
            self.pathfinder.calculate_shortest_path(self.graph, 'A', 'E')

    def test_all_shortest_paths(self):
        """Test calculating all shortest paths between nodes."""
        # Assume all_shortest_paths returns a dict {target: (path, cost)}
        all_paths = self.pathfinder.all_shortest_paths(self.graph, 'A')
        expected = {
            'A': (['A'], 0),
            'B': (['A', 'B'], 1),
            'C': (['A', 'B', 'C'], 3),
            'D': (['A', 'B', 'C', 'D'], 4)
        }
        self.assertEqual(all_paths, expected)

    # TODO: Implement path_exists method in DijkstraPathFinder
    # def test_shortest_path_exists(self):
    #     self.assertTrue(self.pathfinder.path_exists(self.graph, 'A', 'D'))
    #     self.assertTrue(self.pathfinder.path_exists(self.graph, 'B', 'D'))
    #     self.assertFalse(self.pathfinder.path_exists(self.graph, 'D', 'A'))

class TestBellmanFordPathFinder(unittest.TestCase):

    def setUp(self):
        # Graph supports negative weights but no negative cycles
        self.graph = {
            'A': {'B': 4, 'C': 3},
            'B': {'C': -2, 'D': 2},
            'C': {'D': 3},
            'D': {},
        }
        self.pathfinder = BellmanFordPathFinder()

    def test_shortest_path(self):
        path, cost = self.pathfinder.calculate_shortest_path(self.graph, 'A', 'D')
        self.assertEqual(path, ['A', 'B', 'C','D'])
        self.assertEqual(cost, 5)

    def test_negative_weight_handling(self):
        path, cost = self.pathfinder.find_shortest_path(self.graph, 'A', 'C')
        self.assertEqual(path, ['A', 'B', 'C'])
        self.assertEqual(cost, 2)

    def test_path_to_self(self):
        path, cost = self.pathfinder.calculate_shortest_path(self.graph, 'A', 'A')
        self.assertEqual(path, ['A'])
        self.assertEqual(cost, 0)

    def test_unreachable_node(self):
        graph = {
            'A': {'B': 1},
            'B': {},
            'C': {}  # C is isolated
        }
        path, cost = self.pathfinder.calculate_shortest_path(graph, 'A', 'C')
        self.assertIsNone(path)
        self.assertIsNone(cost)

    def test_nonexistent_node(self):
        with self.assertRaises(KeyError):
            self.pathfinder.calculate_shortest_path(self.graph, 'A', 'Z')

        with self.assertRaises(KeyError):
            self.pathfinder.calculate_shortest_path(self.graph, 'Z', 'A')

    def test_negative_cycle_detection(self):
        graph = {
            'A': {'B': 1},
            'B': {'C': -2},
            'C': {'A': -2}  # forms a negative cycle
        }
        path, cost = self.pathfinder.calculate_shortest_path(graph, 'A', 'C')
        self.assertIsNone(path)
        self.assertIsNone(cost)
        
    ### Test for all pairs shortest paths using Bellman-Ford
    def test_all_pairs_shortest_paths(self):
        result = self.pathfinder.calculate_all_shortest_paths(self.graph, self.nodes)

        # Check a few expected results
        self.assertEqual(result['A']['D']['path'], ['A', 'B', 'C', 'D'])
        self.assertEqual(result['A']['D']['distance'], 5)

        self.assertEqual(result['B']['C']['path'], ['B', 'C'])
        self.assertEqual(result['B']['C']['distance'], -2)

        self.assertEqual(result['C']['D']['path'], ['C', 'D'])
        self.assertEqual(result['C']['D']['distance'], 3)

    def test_path_to_self(self):
        result = self.pathfinder.calculate_all_shortest_paths(self.graph, self.nodes)
        for node in self.nodes:
            self.assertEqual(result[node][node]['path'], [node])
            self.assertEqual(result[node][node]['distance'], 0.0)

    def test_unreachable_node_in_all_pairs(self):
        graph = {
            'A': {'B': 1},
            'B': {},
            'C': {}  # C is disconnected
        }
        nodes = ['A', 'B', 'C']
        result = self.pathfinder.calculate_all_shortest_paths(graph, nodes)
        self.assertIsNone(result['A']['C']['path'])
        self.assertIsNone(result['A']['C']['distance'])

    def test_invalid_node(self):
        graph = {
            'A': {'B': 1},
            'B': {},
        }
        nodes = ['A', 'B', 'Z']
        result = self.pathfinder.calculate_all_shortest_paths(graph, nodes)
        self.assertIsNone(result['A']['Z']['path'])
        self.assertIsNone(result['Z']['A']['path'])

    # TODO: Implement path_exists method in BellmanFordPathFinder
    # def test_shortest_path_exists(self):
    #     self.assertTrue(self.pathfinder.path_exists(self.graph, 'A', 'D'))
    #     self.assertTrue(self.pathfinder.path_exists(self.graph, 'B', 'D'))
    #     self.assertFalse(self.pathfinder.path_exists(self.graph, 'D', 'A'))

class TestFindShortestPathDispatch(unittest.TestCase):
    """Test algorithm selection and correctness in find_shortest_path."""

    def test_positive_weights_graph_uses_dijkstra(self):
        graph = {
            'A': {'B': 2, 'C': 5},
            'B': {'C': 1},
            'C': {}
        }
        path, cost, algorithm = find_shortest_path(graph, 'A', 'C')
        self.assertEqual(path, ['A', 'B', 'C'])
        self.assertEqual(cost, 3)
        self.assertEqual(algorithm, 'Dijkstra')

    def test_negative_weights_graph_uses_bellman_ford(self):
        graph = {
            'A': {'B': 4, 'C': 3},
            'B': {'C': -2, 'D': 2},
            'C': {'D': 3},
            'D': {},
        }
        path, cost, algorithm = find_shortest_path(graph, 'A', 'D')
        self.assertEqual(path, ['A', 'B', 'C', 'D'])
        self.assertEqual(cost, 5)
        self.assertEqual(algorithm, 'Bellman-Ford')

    def test_same_source_and_destination(self):
        graph = {
            'A': {'B': 1},
            'B': {'C': 2},
            'C': {}
        }
        path, cost, algorithm = find_shortest_path(graph, 'A', 'A')
        self.assertEqual(path, ['A'])
        self.assertEqual(cost, 0.0)
        # Could be either algorithm, so just check if it's valid
        self.assertIn(algorithm, ['Dijkstra', 'Bellman-Ford'])

    def test_unreachable_node(self):
        graph = {
            'A': {'B': 1},
            'B': {},
            'C': {}  # disconnected
        }
        path, cost, algorithm = find_shortest_path(graph, 'A', 'C')
        self.assertIsNone(path)
        self.assertIsNone(cost)
        self.assertIn(algorithm, ['Dijkstra', 'Bellman-Ford'])

    def test_nonexistent_node(self):
        graph = {
            'A': {'B': 1},
            'B': {}
        }
        with self.assertRaises(ValueError):
            find_shortest_path(graph, 'A', 'Z')
        with self.assertRaises(ValueError):
            find_shortest_path(graph, 'Z', 'A')

if __name__ == '__main__':
    unittest.main()