import heapq
from typing import Dict, List, Tuple, Set, Optional, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)


class DijkstraPathFinder:
    """
    Implementation of Dijkstra's algorithm for shortest path finding.
    """

    def __init__(self):
        """Initialize the Dijkstra path finder."""
        pass

    @staticmethod
    def _validate_non_negative_weights(graph: Dict[str, Dict[str, float]]) -> None:
        """
        Ensure all weights in the graph are non-negative.

        Raises:
            ValueError: If a negative edge weight is found.
        """
        for src, neighbors in graph.items():
            for dest, weight in neighbors.items():
                if weight < 0:
                    raise ValueError(f"Negative weight detected from '{src}' to '{dest}' with weight {weight}")

    @staticmethod
    def calculate_shortest_path(
        graph: Dict[str, Dict[str, float]],
        start: str,
        end: str
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        """
        Calculate the shortest path between two nodes using Dijkstra's algorithm.

        Args:
            graph: A dictionary of dictionaries representing the graph.
            start: Starting node.
            end: Target node.

        Returns:
            A tuple containing the shortest path and its distance.
        """
        DijkstraPathFinder._validate_non_negative_weights(graph)

        if start not in graph or end not in graph:
            logger.warning(f"Start node '{start}' or end node '{end}' not in graph")
            return None, None

        # Initialize distances dictionary with infinity for all nodes except start
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        
        # Keep track of previous nodes to reconstruct the path
        previous = {node: None for node in graph}
        
        # Priority queue with (distance, node)
        queue = [(0, start)]
        # Set to keep track of processed nodes
        processed = set()

        while queue:
            # Get the node with the smallest distance
            current_distance, current_node = heapq.heappop(queue)
            
            # If we've already processed this node, skip it
            if current_node in processed:
                continue
                
            # Mark the node as processed
            processed.add(current_node)
            
            # If we've reached the end node, reconstruct and return the path
            if current_node == end:
                path = []
                while current_node is not None:
                    path.insert(0, current_node)
                    current_node = previous[current_node]
                return path, current_distance
                
            # Check all neighbors of the current node
            for neighbor, weight in graph[current_node].items():
                # Skip if we've already processed this neighbor
                if neighbor in processed:
                    continue
                    
                # Calculate new distance to neighbor
                distance = current_distance + weight
                
                # If we found a better path to the neighbor
                if distance < distances[neighbor]:
                    # Update the distance
                    distances[neighbor] = distance
                    # Remember which node we came from
                    previous[neighbor] = current_node
                    # Add to the priority queue
                    heapq.heappush(queue, (distance, neighbor))
        
        logger.warning(f"No path found from '{start}' to '{end}'")
        return None, None


    @staticmethod
    def calculate_all_shortest_paths(
        graph: Dict[str, Dict[str, float]],
        nodes: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Union[List[str], float]]]]:
        """
        Calculate shortest paths between all pairs of nodes using Dijkstra.

        Args:
            graph: The graph as adjacency list.
            nodes: List of nodes to calculate paths between.

        Returns:
            Dictionary mapping start→end to path and distance.
        """
        DijkstraPathFinder._validate_non_negative_weights(graph)
        result = {}

        for start_node in nodes:
            distances = {node: float('inf') for node in nodes}
            previous = {node: None for node in nodes}
            distances[start_node] = 0
            queue = [(0, start_node)]

            while queue:
                dist, current = heapq.heappop(queue)

                for neighbor, weight in graph.get(current, {}).items():
                    if neighbor not in distances:
                        continue
                    alt = dist + weight
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        previous[neighbor] = current
                        heapq.heappush(queue, (alt, neighbor))

            result[start_node] = {}

            for end_node in nodes:
                if distances[end_node] == float('inf'):
                    result[start_node][end_node] = {'path': None, 'distance': float('inf')}
                    continue

                path = []
                current = end_node
                while current is not None:
                    path.insert(0, current)
                    current = previous[current]

                result[start_node][end_node] = {
                    'path': path,
                    'distance': distances[end_node]
                }

        return result
