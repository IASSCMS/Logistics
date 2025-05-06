import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class BellmanFordPathFinder:
    """
    Bellman-Ford algorithm for graphs with negative weights.
    """
    def __init__(self):
        """Initialize the Bellman-Ford path finder."""
        pass
    
    @staticmethod
    def calculate_shortest_path(
        graph: Dict[str, Dict[str, float]],
        start: str,
        end: str
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        distances = {node: float('inf') for node in graph}
        predecessors = {node: None for node in graph}
        distances[start] = 0

        # Relax edges
        for _ in range(len(graph) - 1):
            for u in graph:
                for v, weight in graph[u].items():
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        predecessors[v] = u

        # Check for negative-weight cycles
        for u in graph:
            for v, weight in graph[u].items():
                if distances[u] + weight < distances[v]:
                    logger.error("Graph contains a negative-weight cycle.")
                    return None, None

        if distances[end] == float('inf'):
            return None, None

        # Reconstruct path
        path = []
        current = end
        while current:
            path.insert(0, current)
            current = predecessors[current]

        return path, distances[end]
    
    @staticmethod
    def calculate_all_shortest_paths(
        graph: Dict[str, Dict[str, float]],
        nodes: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Union[List[str], float]]]]:
        """
        Calculate shortest paths between all pairs of nodes using Bellman-Ford.

        Args:
            graph: A dictionary of dictionaries representing the graph.
            nodes: List of nodes to calculate paths between.

        Returns:
            Dictionary with format:
            {
                start_node: {
                    end_node: {
                        'path': [node1, node2, ...] or None,
                        'distance': total_distance or None
                    }
                }
            }
        """
        result = {}

        for start_node in nodes:
            result[start_node] = {}
            for end_node in nodes:
                if start_node == end_node:
                    result[start_node][end_node] = {
                        'path': [start_node],
                        'distance': 0.0
                    }
                    continue

                try:
                    path, distance = BellmanFordPathFinder.calculate_shortest_path(
                        graph, start_node, end_node
                    )
                    result[start_node][end_node] = {
                        'path': path,
                        'distance': distance
                    }
                except KeyError:
                    result[start_node][end_node] = {
                        'path': None,
                        'distance': None
                    }

        return result
