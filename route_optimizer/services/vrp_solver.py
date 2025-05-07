from typing import Any, Dict, List
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from distance_matrix import Location
from ortools_optimizer import Delivery, Vehicle


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
    time_matrix = (distance_matrix / speed_km_per_hour) * 3600  # hours to seconds
    
    # Create distance and time callbacks
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 100)  # Convert to integer (cm)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node])  # Time in seconds
    
    # Register callbacks
    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Set the cost function (distance)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
    
    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == depot_index:
            return 0
        for delivery in deliveries:
            delivery_index = location_ids.index(delivery.location_id) if delivery.location_id in location_ids else -1
            if from_node == delivery_index:
                return int(delivery.demand * 100)  # Convert to integer (centi-units)
        return 0
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [int(v.capacity * 100) for v in vehicles],
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
                start_seconds = location.time_window_start * 60
                end_seconds = location.time_window_end * 60
                
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
                total_distance += route_distance / 100  # Convert back to km
        
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
