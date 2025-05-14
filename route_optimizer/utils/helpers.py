"""
Helper functions for the route optimizer module.

This module provides various utility functions used across the route optimizer.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import numpy as np
import types
import datetime
import json
from math import radians, cos, sin, asin, sqrt

from route_optimizer.core.distance_matrix import DistanceMatrixBuilder
from route_optimizer.core.constants import MAX_SAFE_DISTANCE, TIME_SCALING_FACTOR

# Set up logging
logger = logging.getLogger(__name__)


def convert_minutes_to_time_str(minutes_from_midnight: int) -> str:
    """
    Convert minutes from midnight to a time string (HH:MM).
    
    Args:
        minutes_from_midnight: Minutes from midnight.
        
    Returns:
        Time string in HH:MM format.
    """
    hours, minutes = divmod(minutes_from_midnight, 60)
    return f"{hours:02d}:{minutes:02d}"


def convert_time_str_to_minutes(time_str: str) -> int:
    """
    Convert a time string (HH:MM) to minutes from midnight.
    
    Args:
        time_str: Time string in HH:MM format.
        
    Returns:
        Minutes from midnight.
    """
    try:
        if not isinstance(time_str, str): # Handle None or other non-string types explicitly
            raise TypeError("Input must be a string.")
            
        parts = time_str.split(':')
        # Enforce HH:MM format by checking lengths of parts
        if len(parts) != 2 or len(parts[0]) != 2 or len(parts[1]) != 2:
            raise ValueError("Input string does not conform to HH:MM format.")
        
        hours, minutes = map(int, parts)
        # Optionally, could add range validation for hours (0-23) and minutes (0-59)
        # if that's a requirement beyond just format. Test "25:00" suggests it's not strict on 23hr max.
        return hours * 60 + minutes
    except (ValueError, AttributeError, TypeError): # Added TypeError for None case
        logger.error(f"Invalid time string format: {time_str}")
        return 0


def format_route_for_display(route: List[str], location_names: Dict[str, str]) -> str:
    """
    Format a route for display, converting location IDs to names.
    
    Args:
        route: List of location IDs in the route.
        location_names: Dictionary mapping location IDs to names.
        
    Returns:
        Formatted route string.
    """
    route_with_names = [f"{location_names.get(loc_id, loc_id)}" for loc_id in route]
    return " → ".join(route_with_names)

def apply_external_factors(
    distance_matrix: np.ndarray,
    time_matrix: np.ndarray,
    external_factors: Dict[Tuple[int, int], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply external factors like traffic or weather to distance and time matrices.
    
    Args:
        distance_matrix: Original distance matrix(in km).
        time_matrix: Original time matrix(in minutes).
        external_factors: Dictionary mapping (i,j) tuples to factors.
                         A factor of 1.0 means no change, >1.0 means slower.
        
    Returns:
        Tuple containing updated distance and time matrices.
    """
    # Create copies to avoid modifying the originals
    updated_distance_matrix = distance_matrix.copy()
    updated_time_matrix = time_matrix.copy()
    
    for (i, j), factor in external_factors.items():
        # Distance doesn't change with traffic/weather, only time does
        updated_time_matrix[i, j] *= factor
    
    return updated_distance_matrix, updated_time_matrix


def detect_isolated_nodes(graph: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Detect nodes in the graph that are isolated (have no outgoing connections).
    
    Args:
        graph: Dictionary representing the graph with format:
              {node1: {node2: distance, ...}, ...}
        
    Returns:
        List of isolated node IDs.
    """
    isolated_nodes = []
    
    all_nodes_in_graph = set(graph.keys())
    nodes_with_incoming_edges = set()
    for source_node, connections in graph.items():
        for target_node in connections:
            nodes_with_incoming_edges.add(target_node)

    for node, connections in graph.items():
        if not connections:  # No outgoing connections
            # Original test implies isolated if no outgoing.
            # If the definition of isolated is "no outgoing AND no incoming", the original was:
            # has_incoming = any(node in neighbors for neighbors in graph.values())
            # if not has_incoming:
            #    isolated_nodes.append(node)
            # To match the test (A and B isolated for graph3), 'A' should be included.
            # 'A' has no outgoing edges.
            isolated_nodes.append(node)
            
    # Handle nodes that might be destinations but not sources (not keys in the graph dict)
    # but this function's signature implies graph.keys() are all nodes to consider.
    # The test implies that being a key in the graph and having no outgoing connections is enough.
    
    return isolated_nodes

def safe_json_dumps(obj: Any) -> str:
    """
    Safely convert an object to a JSON string, handling non-serializable types.
    
    Args:
        obj: Object to convert to JSON.
        
    Returns:
        JSON string representation of the object.
    """
    def handle_non_serializable(o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        # Check if it's a function/method before __dict__ for general objects
        # For most common function types, str(o) is more informative than an empty __dict__
        if callable(o) and not isinstance(o, type): # 'type' is callable (classes), but we want instances or functions
            import types
            if isinstance(o, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
                return str(o) # Let json.dumps add quotes for the string itself
        if hasattr(o, '__dict__'): # For other custom objects that have a useful __dict__
            return o.__dict__
        return str(o) # Fallback: convert to string
    
    return json.dumps(obj, default=handle_non_serializable)



def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Human-readable duration string.
    """
    # TIME_SCALING_FACTOR is expected to be 60 (seconds per minute)
    # For solver's internal representation of time in minutes.
    # The input `seconds` to this function is actual total seconds.
    
    s_int = int(seconds) # Work with integer seconds

    hours, remainder = divmod(s_int, 3600) # 60 * 60
    minutes, secs_remainder = divmod(remainder, 60) # TIME_SCALING_FACTOR
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    
    # Add minutes if hours were shown, or if minutes > 0, or if total is 0s (to show 0m)
    if hours > 0 or minutes > 0:
        parts.append(f"{minutes}m")
    elif s_int == 0: # Special case for exactly 0 seconds total duration
        parts.append("0m")

    # Add seconds if hours or minutes were shown, or if only seconds are non-zero, or if total is 0s
    if hours > 0 or minutes > 0 or secs_remainder > 0:
        parts.append(f"{secs_remainder}s")
    elif s_int == 0: # Ensure "0s" is appended for "0m 0s"
        parts.append("0s")
        
    if not parts: # Should only happen if input seconds was >0 but <60, and hours/minutes logic above didn't add "0m"
                  # e.g., input 30 seconds. h=0,m=0, secs_remainder=30.
                  # parts=[], then parts.append("30s"). Result "30s". Expected "0m 30s".
                  # Let's refine minute part.
        if s_int > 0 and s_int < 3600 : # if less than an hour, ensure minutes are shown
            # This implies the current structure is a bit off.
            # Let's use the logic derived in thought process.
            pass # Will re-write below based on thought process.

    # Corrected logic from thought process:
    s_int = int(seconds)
    hours, remainder = divmod(s_int, 3600)
    minutes, secs_var = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    
    # This logic ensures minutes part is added correctly:
    # - If hours > 0 (e.g., "1h 0m")
    # - If minutes > 0 (e.g., "5m")
    # - If hours is 0 and minutes is 0 (e.g. for "0m 30s" or "0m 0s")
    if hours > 0 or minutes > 0 or (hours == 0 and minutes == 0):
        parts.append(f"{minutes}m")

    # This logic ensures seconds part is added correctly:
    if secs_var > 0 or (parts and secs_var == 0): # if parts already has h or m, and s is 0, add "0s"
        parts.append(f"{secs_var}s")
    
    # If after all this parts is empty, it means input was 0, but "0m" should have been added.
    # The logic for minutes: `if hours > 0 or minutes > 0 or (hours == 0 and minutes == 0):`
    # For 0s input: h=0,m=0. Condition is (F or F or (T and T)) -> T. parts.append("0m").
    # Then for seconds: `secs_var > 0 (F) or (parts(T) and secs_var == 0 (T)) (T)` -> parts.append("0s").
    # Result: "0m 0s". This revised logic appears correct.

    return " ".join(parts)