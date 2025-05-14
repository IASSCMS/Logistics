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
                   Format: {node1: {node2: distance, node3: distance, ...}, ...}
            start: Starting node.
            end: Target node.

        Returns:
            A tuple containing the shortest path (list of nodes) and its
            total distance. Returns (None, None) if no path exists or if
            start/end nodes are not in the graph.
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
                # (This check is valid for Dijkstra with non-negative weights)
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
        Calculate shortest paths between all pairs of specified nodes using Dijkstra.

        This method runs Dijkstra's algorithm starting from each node in the 'nodes'
        list to find the shortest paths to all other nodes in the 'nodes' list.
        The path exploration considers all neighbors available in the main 'graph',
        but the distance and predecessor tracking is scoped to the nodes specified
        in the 'nodes' parameter.

        Args:
            graph: The graph as an adjacency list (dictionary of dictionaries with weights).
                   Example: {'A': {'B': 1, 'C': 4}, 'B': {'A': 1, 'C': 2}}
            nodes: A list of node IDs for which all-pairs shortest paths are to be calculated.
                   Paths will be found from each node in this list to every other node
                   in this list.

        Returns:
            A dictionary where keys are start nodes. Each start node maps to another
            dictionary where keys are end nodes. This inner dictionary contains 'path'
            (a list of nodes) and 'distance' (a float).
            Example:
            {
                'A': {
                    'B': {'path': ['A', 'B'], 'distance': 1.0},
                    'C': {'path': ['A', 'B', 'C'], 'distance': 3.0}
                },
                'B': { ... }
            }
            If a path does not exist between two nodes, 'path' will be None and
            'distance' will be float('inf').
        """
        DijkstraPathFinder._validate_non_negative_weights(graph)
        result = {}

        for start_node in nodes:
            # Initialize distances and previous nodes for the current start_node,
            # considering only the nodes specified in the 'nodes' list for these data structures.
            distances = {node: float('inf') for node in nodes}
            previous = {node: None for node in nodes}
            
            if start_node not in graph: # If start_node itself isn't in the main graph
                result[start_node] = {end_node: {'path': None, 'distance': float('inf')} for end_node in nodes}
                if start_node in nodes: # if it was a target node for itself
                     result[start_node][start_node] = {'path': [start_node] if start_node in graph else None, 'distance': 0.0 if start_node in graph else float('inf')}
                continue

            distances[start_node] = 0
            queue = [(0, start_node)] # Priority queue: (distance, node)

            while queue:
                dist, current = heapq.heappop(queue)

                # Optimization: If we've found a shorter path to 'current' already
                # after this entry was added to the queue, skip processing this stale entry.
                if dist > distances[current]:
                    continue

                # Explore neighbors from the main 'graph' definition
                for neighbor, weight in graph.get(current, {}).items():
                    # Only consider neighbors that are part of the specified 'nodes' list
                    # for distance updates and path construction.
                    if neighbor not in distances: 
                        continue

                    alt = dist + weight
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        previous[neighbor] = current
                        heapq.heappush(queue, (alt, neighbor))

            # Store results for the current start_node
            result[start_node] = {}
            for end_node in nodes:
                if distances[end_node] == float('inf'):
                    result[start_node][end_node] = {'path': None, 'distance': float('inf')}
                    continue

                path = []
                curr_path_node = end_node
                while curr_path_node is not None:
                    path.insert(0, curr_path_node)
                    curr_path_node = previous[curr_path_node]
                
                # Ensure the reconstructed path actually starts with start_node if a path was found
                if path and path[0] == start_node:
                    result[start_node][end_node] = {
                        'path': path,
                        'distance': distances[end_node]
                    }
                elif start_node == end_node and distances[end_node] == 0: # Path to self
                     result[start_node][end_node] = {'path': [start_node], 'distance': 0.0}
                else: # Path reconstruction failed or inconsistent
                    result[start_node][end_node] = {'path': None, 'distance': float('inf')}
                    if start_node == end_node : # Special case for self-path if start_node not in graph but in nodes
                         result[start_node][end_node] = {'path': None, 'distance': float('inf')}


        return result