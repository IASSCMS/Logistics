from typing import Any, Dict, List
from venv import logger
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from route_optimizer.core.constants import CAPACITY_SCALING_FACTOR, DISTANCE_SCALING_FACTOR, MAX_SAFE_DISTANCE, MAX_SAFE_TIME, TIME_SCALING_FACTOR
from route_optimizer.core.types_1 import Location
from route_optimizer.models import Vehicle, Delivery


def solve_with_time_windows(
    self,
    distance_matrix: np.ndarray,
    location_ids: List[str],
    vehicles: List[Vehicle],
    deliveries: List[Delivery],
    locations: List[Location],
    depot_index: int = 0,
    speed_km_per_hour: float = 50.0
) -> Dict[str, Any]:
    """
    Solve the Vehicle Routing Problem with Time Windows.
    
    Args:
        distance_matrix: Matrix of distances between locations.
        location_ids: List of location IDs corresponding to the distance matrix.
        vehicles: List of available vehicles.
        deliveries: List of deliveries to be made.
        locations: List of locations with time window information.
        depot_index: Index of the depot in the location list.
        speed_km_per_hour: Average vehicle speed in km/h for time calculations.
        
    Returns:
        Dictionary containing the solution details.
    """
    # Create the routing model
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        len(vehicles),
        depot_index
    )
    routing = pywrapcp.RoutingModel(manager)
    
    # Convert distance to travel time (in seconds)
    time_matrix = (distance_matrix / speed_km_per_hour) * 60 * TIME_SCALING_FACTOR  # hours to seconds

    # Create a mapping from location indices to Location objects
    location_index_to_location = {index: location for index, location in enumerate(locations)}
    
    def distance_callback(from_index, to_index):
        """Returns the scaled distance between the two nodes."""
        try:
            # Convert from routing variable Index to distance matrix NodeIndex
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # Get the raw distance value
            raw_distance = distance_matrix[from_node][to_node]
            
            # Check if it's a valid number and not too large
            if np.isinf(raw_distance) or np.isnan(raw_distance):
                logger.warning(f"Invalid distance value {raw_distance} from node {from_node} to {to_node}")
                return int(MAX_SAFE_DISTANCE * DISTANCE_SCALING_FACTOR)
                
            # Apply scaling with bounds checking
            safe_distance = min(raw_distance, MAX_SAFE_DISTANCE)
            scaled_distance = int(safe_distance * DISTANCE_SCALING_FACTOR)
            
            return scaled_distance
        except Exception as e:
            logger.error(f"Error in distance callback for indices: {from_index}, {to_index}: {str(e)}")
            # Return a large but valid distance as fallback
            return int(MAX_SAFE_DISTANCE * DISTANCE_SCALING_FACTOR)
    
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        try:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # If using a pre-calculated time matrix
            if 'time_matrix' in locals() or 'time_matrix' in globals():
                raw_time = time_matrix[from_node][to_node]
                if np.isinf(raw_time) or np.isnan(raw_time):
                    return int(MAX_SAFE_TIME * TIME_SCALING_FACTOR)
                return int(raw_time)  # Assume time_matrix already has values in seconds
            
            # Otherwise calculate from distance
            distance_km = distance_matrix[from_node][to_node]
            if np.isinf(distance_km) or np.isnan(distance_km):
                distance_km = MAX_SAFE_DISTANCE
            
            travel_minutes = (min(distance_km, MAX_SAFE_DISTANCE) / speed_km_per_hour) * 60
            to_loc = location_index_to_location.get(to_node)
            service_time = to_loc.service_time if to_loc else 0
            
            total_time_seconds = (travel_minutes + service_time) * TIME_SCALING_FACTOR
            return int(total_time_seconds)
        except Exception as e:
            logger.error(f"Error in time callback: {str(e)}")
            return int(MAX_SAFE_TIME * TIME_SCALING_FACTOR)
    
    # Register callbacks
    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Set the cost function (distance)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
    
    # Add capacity constraints
    def demand_callback(from_index):
        """Returns the scaled demand for the node."""
        try:
            from_node = manager.IndexToNode(from_index)
            location_id = location_ids[from_node]
            
            # Find all deliveries at this location
            total_demand = 0
            for delivery in deliveries:
                if delivery.location_id == location_id:
                    # Add pickups as negative demand, deliveries as positive
                    demand_value = -delivery.demand if delivery.is_pickup else delivery.demand
                    total_demand += demand_value
            
            # Apply scaling with bounds checking
            logger.debug(f"Raw demand at node {from_node}: {total_demand}")
            scaled_demand = int(total_demand * CAPACITY_SCALING_FACTOR)
            logger.debug(f"Scaled demand: {scaled_demand}")
            return scaled_demand
        except Exception as e:
            logger.error(f"Error in demand callback for index {from_index}: {str(e)}")
            return 0
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [int(v.capacity * CAPACITY_SCALING_FACTOR) for v in vehicles],
        True,
        'Capacity'
    )
    
    # Time Dimension
    routing.AddDimension(
        time_callback_index,
        3600,  # wait time allowed (1 hour in seconds)
        86400,  # max time (24hr) per vehicle
        False,
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Add time window constraints
    for location_idx, location in enumerate(locations):
        if hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end'):
            if location.time_window_start is not None and location.time_window_end is not None:
                # Convert minutes to seconds
                start_seconds = location.time_window_start * TIME_SCALING_FACTOR
                end_seconds = location.time_window_end * TIME_SCALING_FACTOR
                
                for vehicle_idx in range(len(vehicles)):
                    index = manager.NodeToIndex(location_idx)
                    time_dimension.CumulVar(index).SetRange(start_seconds, end_seconds)
    
    # Add vehicle start and end time constraints
    for vehicle_id, vehicle in enumerate(vehicles):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetValue(0)  # Start at time 0
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = self.time_limit_seconds
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Return the solution
    if solution:
        routes = []
        assigned_vehicles = {}
        total_distance = 0
        delivery_locations = set()
        arrival_times = {}
        
        for vehicle_id in range(len(vehicles)):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            arrival_time_list = []
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location_id = location_ids[node_index]
                
                # Get arrival time
                arrival_time = solution.Value(time_dimension.CumulVar(index))
                arrival_time_list.append(arrival_time)
                
                if node_index != depot_index:
                    delivery_locations.add(location_id)
                
                route.append(location_id)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            # Add the depot back as the end point
            node_index = manager.IndexToNode(index)
            location_id = location_ids[node_index]
            arrival_time = solution.Value(time_dimension.CumulVar(index))
            arrival_time_list.append(arrival_time)
            route.append(location_id)
            
            # Don't add empty routes
            if len(route) > 2:  # More than just start-end depot
                routes.append(route)
                assigned_vehicles[vehicles[vehicle_id].id] = len(routes) - 1
                arrival_times[len(routes) - 1] = arrival_time_list
                total_distance += route_distance / DISTANCE_SCALING_FACTOR
        
        # Check for unassigned deliveries
        unassigned_deliveries = []
        for delivery in deliveries:
            if delivery.location_id not in delivery_locations:
                unassigned_deliveries.append(delivery.id)
        
        return {
            'status': 'success',
            'routes': routes,
            'total_distance': total_distance,
            'assigned_vehicles': assigned_vehicles,
            'unassigned_deliveries': unassigned_deliveries,
            'arrival_times': arrival_times
        }
    else:
        # No solution found
        return {
            'status': 'failure',
            'routes': [],
            'total_distance': 0,
            'assigned_vehicles': {},
            'unassigned_deliveries': [delivery.id for delivery in deliveries]
        }
