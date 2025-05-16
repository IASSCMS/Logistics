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
        nodes_subset: List[str]  # Renamed for clarity
    ) -> Dict[str, Dict[str, Dict[str, Union[List[str], float]]]]:
        """
        Calculate shortest paths between all pairs of specified nodes using Dijkstra.

        This method runs Dijkstra's algorithm starting from each node in the 'nodes_subset'
        list to find the shortest paths to all other nodes in the 'nodes_subset' list.
        The path exploration considers all neighbors and intermediate nodes available 
        in the main 'graph'.

        Args:
            graph: The graph as an adjacency list.
            nodes_subset: A list of node IDs for which all-pairs shortest paths are calculated.

        Returns:
            A dictionary structured as {start_node: {end_node: {'path': [], 'distance': 0.0}}}.
            If a path does not exist, 'path' is None and 'distance' is float('inf').
        """
        DijkstraPathFinder._validate_non_negative_weights(graph)
        result = {}

        # Pre-initialize the result structure for all pairs in nodes_subset
        for s_node in nodes_subset:
            result[s_node] = {}
            for e_node in nodes_subset:
                if s_node == e_node:
                    # Path to self is 0 if node is in graph, otherwise inf
                    result[s_node][e_node] = {
                        'path': [s_node] if s_node in graph else None,
                        'distance': 0.0 if s_node in graph else float('inf')
                    }
                else:
                    result[s_node][e_node] = {'path': None, 'distance': float('inf')}

        for start_node in nodes_subset:
            if start_node not in graph:
                # If start_node is not in the graph, all paths from it remain None/inf
                # (already handled by pre-initialization for pairs involving this start_node)
                continue

            # Dijkstra's from start_node to ALL nodes in the full graph
            # These dictionaries are for the current Dijkstra run, keyed by all nodes in the graph
            current_run_distances = {node_in_graph: float('inf') for node_in_graph in graph}
            current_run_previous = {node_in_graph: None for node_in_graph in graph}
            
            current_run_distances[start_node] = 0
            priority_queue = [(0, start_node)]  # (distance, node_in_graph)
            
            processed_nodes_in_run = set()

            while priority_queue:
                dist, current_g_node = heapq.heappop(priority_queue)

                if current_g_node in processed_nodes_in_run:
                    continue
                processed_nodes_in_run.add(current_g_node)

                # Optimization: if a shorter path to current_g_node was found after this entry was queued
                if dist > current_run_distances[current_g_node]:
                    continue

                for neighbor_g, weight in graph.get(current_g_node, {}).items():
                    # neighbor_g must be in the graph for a valid edge
                    if neighbor_g not in graph: # Should not happen if graph is well-formed
                        logger.warning(f"Neighbor {neighbor_g} of {current_g_node} not found in graph keys.")
                        continue

                    alt_dist = dist + weight
                    if alt_dist < current_run_distances[neighbor_g]:
                        current_run_distances[neighbor_g] = alt_dist
                        current_run_previous[neighbor_g] = current_g_node
                        heapq.heappush(priority_queue, (alt_dist, neighbor_g))
            
            # After Dijkstra from start_node, populate results for relevant end_nodes
            for end_node in nodes_subset:
                if end_node not in graph: # Target end_node not in graph
                    # result[start_node][end_node] already initialized to None/inf
                    continue

                final_dist_to_end_node = current_run_distances.get(end_node, float('inf'))

                if final_dist_to_end_node == float('inf'):
                    # Path to self was pre-initialized; other non-existent paths also pre-initialized
                    if start_node == end_node and start_node in graph: # ensure path to self is correctly 0 if node in graph
                         result[start_node][end_node] = {'path': [start_node], 'distance': 0.0}
                    else:
                        result[start_node][end_node] = {'path': None, 'distance': float('inf')}
                    continue

                # Reconstruct path
                path = []
                path_tracer_node = end_node
                while path_tracer_node is not None:
                    path.insert(0, path_tracer_node)
                    if path_tracer_node == start_node: # Reached the start of the path
                        break
                    
                    predecessor = current_run_previous.get(path_tracer_node)
                    if predecessor is None and path_tracer_node != start_node:
                        # Should not happen if final_dist_to_end_node is not 'inf'
                        logger.error(f"Path reconstruction error: No predecessor for {path_tracer_node} "
                                     f"from {start_node} to {end_node}.")
                        path = [] # Invalidate path
                        break
                    path_tracer_node = predecessor
                
                # Validate reconstructed path
                if path and path[0] == start_node and (len(path) == 1 or path[-1] == end_node) :
                    result[start_node][end_node] = {'path': path, 'distance': final_dist_to_end_node}
                elif start_node == end_node and final_dist_to_end_node == 0: # Correct path to self
                    result[start_node][end_node] = {'path': [start_node], 'distance': 0.0}
                else: # Path reconstruction failed or other inconsistency
                    logger.warning(f"Path reconstruction to {end_node} from {start_node} resulted in inconsistent path: {path} "
                                   f"with distance {final_dist_to_end_node}. Setting to None/inf.")
                    result[start_node][end_node] = {'path': None, 'distance': float('inf')}
        return result
