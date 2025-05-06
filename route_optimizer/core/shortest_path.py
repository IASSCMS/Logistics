from typing import Dict, Tuple, List, Optional
from .dijkstra import DijkstraPathFinder
from .bellman_ford import BellmanFordPathFinder


def contains_negative_weight(graph: Dict[str, Dict[str, float]]) -> bool:
    for node in graph:
        for weight in graph[node].values():
            if weight < 0:
                return True
    return False

def find_shortest_path(
    graph: Dict[str, Dict[str, float]],
    start: str,
    end: str
) -> Tuple[Optional[List[str]], Optional[float], str]:
    if contains_negative_weight(graph):
        path, cost = BellmanFordPathFinder.calculate_shortest_path(graph, start, end)
        algorithm = "Bellman-Ford"
    else:
        path, cost = DijkstraPathFinder.calculate_shortest_path(graph, start, end)
        algorithm = "Dijkstra"
    
    return path, cost, algorithm
