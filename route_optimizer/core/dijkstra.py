"""
Implementation of Dijkstra's algorithm for shortest path finding.

This module provides functions to calculate the shortest path between locations
using Dijkstra's algorithm.
"""
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
    def calculate_shortest_path(
        graph: Dict[str, Dict[str, float]],
        start: str,
        end: str
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        """
        Calculate the shortest path between two nodes using Dijkstra's algorithm.
        """
        # Check if start and end nodes are in the graph
        if start not in graph or end not in graph:
            logger.warning(f"Start node '{start}' or end node '{end}' not in graph")
            return None, None

        # Handle the case where start and end are the same
        if start == end:
            return [start], 0.0
            
        # Check for negative weights - for test compatibility, we'll just log and continue
        for node in graph:
            for neighbor, weight in graph[node].items():
                if weight < 0:
                    logger.error(f"Negative weight from {node} to {neighbor}. Dijkstra cannot proceed.")
                    return None, None
                
        # Initialize distances and predecessors
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        predecessors = {node: None for node in graph}
        
        # Priority queue: (distance, node)
        queue = [(0, start)]
        visited = set()

        while queue:
            current_distance, current = heapq.heappop(queue)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Found the end node
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                return path, distances[end]
            
            # Explore neighbors
            for neighbor, weight in graph[current].items():
                if neighbor not in graph:
                    logger.warning(f"Neighbor {neighbor} of node {current} not found in graph.")
                    continue
                    
                distance = current_distance + weight
                
                # Only update if we found a shorter path
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(queue, (distance, neighbor))
        
        # If we get here, no path was found
        logger.warning(f"No path found from '{start}' to '{end}'")
        return None, None


    @staticmethod
    def calculate_all_shortest_paths(
        graph: Dict[str, Dict[str, float]],
        nodes: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Union[List[str], float]]]]:
        """
        Calculate shortest paths between all pairs of nodes.

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
                
                path, distance = DijkstraPathFinder.calculate_shortest_path(
                    graph, start_node, end_node
                )
                
                result[start_node][end_node] = {
                    'path': path,
                    'distance': distance
                }
        
        return result