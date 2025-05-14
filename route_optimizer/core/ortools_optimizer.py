"""
Implementation of route optimization using Google OR-Tools.

This module provides classes and functions for solving Vehicle Routing Problems
(VRP) using Google's OR-Tools library.
"""
import logging
import numpy as np

from dataclasses import dataclass, field
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from typing import Dict, List, Tuple, Optional, Any
from route_optimizer.core.constants import CAPACITY_SCALING_FACTOR, DISTANCE_SCALING_FACTOR, MAX_SAFE_DISTANCE, MAX_SAFE_TIME, TIME_SCALING_FACTOR
from route_optimizer.core.types_1 import Location, OptimizationResult, validate_optimization_result
from route_optimizer.models import Vehicle, Delivery

MAX_ROUTE_DURATION_UNSCALED = 24 * 60 # e.g., 24 hours in minutes, for dimension capacity
MAX_ROUTE_DISTANCE_UNSCALED = 5000 # e.g., 5000 km, for dimension capacity
COST_COEFFICIENT_FOR_LOAD_BALANCE = 100 # Tunable: Higher values prioritize balancing more

# Set up logging
logger = logging.getLogger(__name__)

class ORToolsVRPSolver:
    """
    Vehicle Routing Problem solver using Google OR-Tools.
    """

    def __init__(self, time_limit_seconds: int = 30):
        """
        Initialize the VRP solver.

        Args:
            time_limit_seconds: Time limit for the solver in seconds.
        """
        self.time_limit_seconds = time_limit_seconds

    def solve(
        self,
        distance_matrix: np.ndarray,
        location_ids: List[str],
        vehicles: List[Vehicle],
        deliveries: List[Delivery],
        depot_index: int = 0,
        time_matrix: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Solve the Vehicle Routing Problem using OR-Tools.

        Args:
            distance_matrix: Matrix of distances between locations.
            location_ids: List of location IDs corresponding to matrix indices.
            vehicles: List of Vehicle objects with capacity and constraints.
            deliveries: List of Delivery objects with demands and constraints.
            depot_index: Index of the depot in the distance matrix.
            consider_time_windows: Whether to consider time windows in optimization.

        Returns:
            OptimizationResult containing:
                - status: 'success' or 'failed'
                - routes: List of routes, each a list of location IDs
                - total_distance: Sum of all route distances
                - total_cost: Total cost accounting for distance and fixed costs
                - assigned_vehicles: Map of vehicle IDs to route indices
                - unassigned_deliveries: List of deliveries that couldn't be assigned
                - detailed_routes: Empty list (populated by subsequent processing)
                - statistics: Basic statistics about the solution
        """
        # Create the routing index manager
        num_locations = len(location_ids)
        num_vehicles = len(vehicles)
        
        # Create location_id to index mapping
        location_id_to_index = {loc_id: idx for idx, loc_id in enumerate(location_ids)}
        
        # Map vehicle start/end locations to indices
        starts = []
        ends = []
        
        # Special case: No deliveries, create direct depot-to-depot routes
        if not deliveries:
            routes = []
            assigned_vehicles = {}
            
            # For each vehicle, create a direct depot-to-depot route
            for idx, vehicle in enumerate(vehicles):
                # Get the start and end location IDs
                start_location_id = vehicle.start_location_id
                end_location_id = vehicle.end_location_id or vehicle.start_location_id
                
                # Create a direct route from start to end
                route = [start_location_id, end_location_id]
                routes.append(route)
                assigned_vehicles[vehicle.id] = idx
            
            return OptimizationResult(
                status='success',
                routes=routes,
                total_distance=0.0,  # Since we're not calculating actual distances
                total_cost=0.0,      # No cost for empty routes
                assigned_vehicles=assigned_vehicles,
                unassigned_deliveries=[],
                detailed_routes=[],
                statistics={'info': 'Empty problem: direct depot-to-depot routes created'}
            )
            
        for vehicle in vehicles:
            try:
                start_idx = location_id_to_index[vehicle.start_location_id]
                # If end location not specified, use the start location
                end_idx = location_id_to_index.get(
                    vehicle.end_location_id or vehicle.start_location_id,
                    start_idx
                )
                starts.append(start_idx)
                ends.append(end_idx)
            except KeyError as e:
                logger.error(f"Vehicle location not found in locations: {e}")
                return OptimizationResult(
                    status='failed',
                    routes=[],
                    total_distance=0.0,
                    total_cost=0.0,
                    assigned_vehicles={},
                    unassigned_deliveries=[d.id for d in deliveries],
                    detailed_routes=[],
                    statistics={'error': f"Vehicle location not found: {e}"}
                )
        
        manager = pywrapcp.RoutingIndexManager(
            num_locations, 
            num_vehicles,
            starts,
            ends
        )
        
        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register a transit callback
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
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add Capacity constraint
        def demand_callback(from_index):
            """Returns the scaled demand for the node."""
            try:
                from_node = manager.IndexToNode(from_index)
                location_id = location_ids[from_node]
                
                # Find all deliveries at this location
                total_demand = 0
                for delivery in deliveries:
                    if delivery.location_id == location_id:
                        # Handle both size and demand attributes for compatibility
                        demand_value = delivery.demand if hasattr(delivery, 'demand') else delivery.size
                        # Add pickups as negative demand, deliveries as positive
                        if hasattr(delivery, 'is_pickup') and delivery.is_pickup:
                            demand_value = -demand_value
                        total_demand += demand_value
                
                # Apply scaling
                return int(total_demand * CAPACITY_SCALING_FACTOR)
            except Exception as e:
                logger.error(f"Error in demand callback for index {from_index}: {str(e)}")
                return 0
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [int(v.capacity * CAPACITY_SCALING_FACTOR) for v in vehicles],  # Make sure the capacity is converted to integer
            True,
            'Capacity'
        )
        
        # --- START: Load Balancing Logic ---
        # Choose dimension for balancing: time if time_matrix is available, otherwise distance.
        if time_matrix is not None:
            logger.info("Attempting to balance load by route TTIME.")
            dimension_name = 'RouteTime'
            
            # Create a transit callback for time (travel time only from time_matrix)
            def time_balance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                # Assuming time_matrix contains travel times in a unit that needs scaling (e.g., minutes)
                # Adjust scaling and units as per your time_matrix content.
                # This example assumes time_matrix values are, e.g., in minutes.
                raw_time = time_matrix[from_node][to_node]
                if np.isinf(raw_time) or np.isnan(raw_time):
                    # Fallback for invalid time values
                    return int(MAX_ROUTE_DURATION_UNSCALED * TIME_SCALING_FACTOR) # A large, safe time
                # Scale time for OR-Tools (it prefers integers)
                return int(raw_time * TIME_SCALING_FACTOR)

            time_balance_transit_callback_index = routing.RegisterTransitCallback(time_balance_callback)
            
            routing.AddDimension(
                time_balance_transit_callback_index,
                0,  # No slack
                int(MAX_ROUTE_DURATION_UNSCALED * TIME_SCALING_FACTOR),  # Max total time per vehicle (scaled)
                True,  # Start cumul to zero
                dimension_name
            )
        else:
            logger.info("Attempting to balance load by route DISTANCE (time_matrix not provided).")
            dimension_name = 'RouteDistance'
            
            # We can reuse the existing distance_callback logic for consistency if it represents travel cost/effort well.
            # The distance_callback already returns scaled distances.
            # Note: transit_callback_index is already defined from the main distance callback for arc costs.
            # We use the same callback for accumulating distance for balancing.
            
            routing.AddDimension(
                transit_callback_index, # Re-using the main distance callback
                0,  # No slack
                int(MAX_ROUTE_DISTANCE_UNSCALED * DISTANCE_SCALING_FACTOR),  # Max total distance per vehicle (scaled)
                True,  # Start cumul to zero
                dimension_name
            )

        balancing_dimension = routing.GetDimensionOrDie(dimension_name)
        # This makes the solver try to minimize the maximum of the end cumul vars (total time/distance)
        # for the specified dimension across all vehicles. A higher coefficient prioritizes balancing.
        balancing_dimension.SetGlobalSpanCostCoefficient(COST_COEFFICIENT_FOR_LOAD_BALANCE)
        logger.info(f"Set GlobalSpanCostCoefficient on '{dimension_name}' dimension for load balancing with coefficient {COST_COEFFICIENT_FOR_LOAD_BALANCE}.")
        # --- END: Load Balancing Logic ---
        
        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = self.time_limit_seconds # Use the instance variable
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        # Process the solution
        if solution:
            routes_list = []
            total_distance_val = 0 
            assigned_vehicles_map = {}
            
            for vehicle_idx in range(num_vehicles):
                route_for_vehicle = []
                index = routing.Start(vehicle_idx)
                current_route_distance = 0
                while not routing.IsEnd(index):
                    node_idx = manager.IndexToNode(index)
                    route_for_vehicle.append(location_ids[node_idx])
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    # Accumulate distance for this vehicle's route
                    current_route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_idx
                    )
                
                node_idx = manager.IndexToNode(index) # Add end node
                route_for_vehicle.append(location_ids[node_idx])
                
                # Only add route if it's meaningful (more than just depot to depot if empty, or has actual stops)
                if len(route_for_vehicle) > 2 or (len(route_for_vehicle) == 2 and route_for_vehicle[0] != route_for_vehicle[1]): 
                    routes_list.append(route_for_vehicle)
                    assigned_vehicles_map[vehicles[vehicle_idx].id] = len(routes_list) - 1
                    total_distance_val += (current_route_distance / DISTANCE_SCALING_FACTOR) # Sum up scaled back distances
            
            delivery_locations = set()
            for r_list in routes_list: 
                delivery_locations.update(r_list)
            
            unassigned_deliveries_ids = [ 
                d.id for d in deliveries if d.location_id not in delivery_locations
            ]
            
            result = OptimizationResult(
                status='success',
                routes=routes_list, 
                total_distance=total_distance_val, 
                total_cost=0.0,
                assigned_vehicles=assigned_vehicles_map, 
                unassigned_deliveries=unassigned_deliveries_ids, 
                detailed_routes=[],
                statistics={}
            )
        else:
            result = OptimizationResult(
                status='failed', routes=[], total_distance=0.0, total_cost=0.0,
                assigned_vehicles={}, unassigned_deliveries=[d.id for d in deliveries],
                detailed_routes=[], statistics={'error': 'No solution found!'}
            )
        
        return result
        
    def solve_with_time_windows(
        self,
        distance_matrix: np.ndarray,
        location_ids: List[str],
        vehicles: List[Vehicle],
        deliveries: List[Delivery],
        locations: List[Location], 
        depot_index: int = 0,
        speed_km_per_hour: float = 50.0
    ) -> OptimizationResult: 
        """
        Solve the Vehicle Routing Problem with Time Windows.

        Returns:
            OptimizationResult object containing the solution details. # CHANGED docstring
        """
        num_locations = len(location_ids)
        num_vehicles = len(vehicles)

        # Mappings
        location_id_to_index = {loc_id: idx for idx, loc_id in enumerate(location_ids)}
        location_index_to_location = {
            idx: next((loc for loc in locations if loc.id == loc_id), None)
            for idx, loc_id in enumerate(location_ids)
        }

        # Set up start and end indices
        starts = []
        ends = []

        for vehicle in vehicles:
            try:
                start_idx = location_id_to_index[vehicle.start_location_id]
                end_idx = location_id_to_index.get(vehicle.end_location_id or vehicle.start_location_id, start_idx)
                starts.append(start_idx)
                ends.append(end_idx)
            except KeyError as e:
                logger.error(f"Vehicle location not found in locations: {e}")
                return OptimizationResult(
                    status='failed',
                    routes=[],
                    total_distance=0.0,
                    total_cost=0.0,
                    assigned_vehicles={},
                    unassigned_deliveries=[d.id for d in deliveries],
                    detailed_routes=[],
                    statistics={'error': f"Vehicle location not found: {e}"}
                )

        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
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

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def time_callback(from_index, to_index):
            """Returns the total scaled travel time (travel + service) between the two nodes in seconds."""
            try:
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)

                # 1. Calculate travel time in minutes
                distance_km = distance_matrix[from_node][to_node]
                    
                # Handle invalid distances
                if np.isinf(distance_km) or np.isnan(distance_km):
                    logger.warning(f"Invalid distance for time_callback from {from_node} to {to_node}. Using MAX_SAFE_DISTANCE.")
                    distance_km = MAX_SAFE_DISTANCE 
                    
                # Ensure distance_km is capped if it's excessively large but valid number
                safe_distance_km = min(distance_km, MAX_SAFE_DISTANCE)
                travel_minutes = (safe_distance_km / speed_km_per_hour) * 60

                # 2. Get service time for the destination node (to_node) in minutes
                to_loc_object = location_index_to_location.get(to_node)
                service_time_minutes = 0
                if to_loc_object and hasattr(to_loc_object, 'service_time') and to_loc_object.service_time is not None:
                    service_time_minutes = to_loc_object.service_time
                else:
                    # Depot or location without service time
                    pass

                # 3. Total time in minutes
                total_minutes = travel_minutes + service_time_minutes
                    
                # 4. Scale total time to seconds using TIME_SCALING_FACTOR
                # TIME_SCALING_FACTOR = 60 (converts minutes to seconds)
                total_time_seconds_scaled = int(total_minutes * TIME_SCALING_FACTOR)
                    
                # Ensure the returned value is within a safe bound for OR-Tools (e.g., related to MAX_SAFE_TIME)
                # MAX_SAFE_TIME from constants is in minutes.
                max_solver_time = int(MAX_SAFE_TIME * TIME_SCALING_FACTOR) # Max safe time in scaled seconds

                return min(total_time_seconds_scaled, max_solver_time)

            except Exception as e:
                logger.error(f"Error in time_callback from {from_index} to {to_index}: {str(e)}", exc_info=True)
                # Fallback to a large, scaled time value (MAX_SAFE_TIME in minutes, scaled to seconds)
                return int(MAX_SAFE_TIME * TIME_SCALING_FACTOR)

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        # Time Dimension
        routing.AddDimension(
            time_callback_index,
            3600,  # wait time allowed
            86400,  # max time (24hr) per vehicle
            False,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # Add time windows to each location
        for idx, location_id_iter in enumerate(location_ids):
            loc = location_index_to_location.get(idx)
            if loc and loc.time_window_start is not None and loc.time_window_end is not None:
                # Assuming time_window_start/end are in minutes, and TIME_SCALING_FACTOR converts minutes to seconds
                start_tw = loc.time_window_start * TIME_SCALING_FACTOR 
                end_tw = loc.time_window_end * TIME_SCALING_FACTOR
                node_manager_index = manager.NodeToIndex(idx)
                time_dimension.CumulVar(node_manager_index).SetRange(start_tw, end_tw)

        # Add capacity constraints
        def demand_callback(from_index):
            """Returns the scaled demand for the node."""
            try:
                from_node = manager.IndexToNode(from_index)
                current_location_id = location_ids[from_node]
                
                # Find all deliveries at this location
                total_demand = 0
                for delivery in deliveries:
                    if delivery.location_id == current_location_id:
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

        # --- START: Load Balancing Logic ---
        # We will balance based on the existing 'Time' dimension, which includes
        # travel time, service time, and waiting time.
        logger.info("Attempting to balance load by total route duration (Time dimension).")
        # The 'Time' dimension is already created and configured for time windows.
        # time_dimension = routing.GetDimensionOrDie('Time') # Already fetched above
        time_dimension.SetGlobalSpanCostCoefficient(COST_COEFFICIENT_FOR_LOAD_BALANCE)
        logger.info(f"Set GlobalSpanCostCoefficient on 'Time' dimension for load balancing with coefficient {COST_COEFFICIENT_FOR_LOAD_BALANCE}.")
        # --- END: Load Balancing Logic ---
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = self.time_limit_seconds

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            processed_routes = [] # Will store lists of location IDs
            detailed_routes_data = [] # Will store list of dicts for DetailedRoute
            assigned_vehicles_map = {}
            total_distance_val = 0.0
            all_visited_location_ids = set() # To find unassigned deliveries

            for vehicle_idx in range(num_vehicles):
                route_stops_ids = []
                route_detailed_stops = [] # For arrival times
                index = routing.Start(vehicle_idx)
                
                current_route_distance = 0
                
                while not routing.IsEnd(index):
                    node_idx = manager.IndexToNode(index)
                    loc_id = location_ids[node_idx]
                    route_stops_ids.append(loc_id)
                    all_visited_location_ids.add(loc_id)

                    time_var = time_dimension.CumulVar(index)
                    arrival_time_seconds = solution.Min(time_var)
                    route_detailed_stops.append({
                        'location_id': loc_id,
                        'arrival_time_seconds': arrival_time_seconds 
                        # You might want to convert arrival_time_seconds to minutes 
                        # if OptimizationResult expects that for its stats/detailed_routes
                    })
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    current_route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)

                # Add the end location
                node_idx = manager.IndexToNode(index)
                loc_id_end = location_ids[node_idx]
                route_stops_ids.append(loc_id_end)
                all_visited_location_ids.add(loc_id_end) # Though end depot might not be a delivery loc
                
                time_var_end = time_dimension.CumulVar(index)
                arrival_time_seconds_end = solution.Min(time_var_end)
                route_detailed_stops.append({
                    'location_id': loc_id_end,
                    'arrival_time_seconds': arrival_time_seconds_end
                })

                # Only add non-empty routes (routes that visit more than just start/end depot if they are the same)
                # Or if start and end are different, a route with just two stops is valid.
                # A simple check: if it has intermediate stops or if start != end.
                is_meaningful_route = False
                if len(route_stops_ids) > 2: # Depot -> Stop -> Depot
                    is_meaningful_route = True
                elif len(route_stops_ids) == 2 and route_stops_ids[0] != route_stops_ids[1]: # DepotA -> DepotB
                     is_meaningful_route = True
                elif not deliveries: # If no deliveries, depot-to-depot is fine
                    is_meaningful_route = True


                if is_meaningful_route:
                    processed_routes.append(route_stops_ids)
                    route_total_distance = current_route_distance / DISTANCE_SCALING_FACTOR
                    total_distance_val += route_total_distance
                    
                    assigned_vehicles_map[vehicles[vehicle_idx].id] = len(processed_routes) - 1
                    
                    # Store arrival times per location for this route
                    estimated_arrival_times_dict = {
                        stop_info['location_id']: stop_info['arrival_time_seconds'] 
                        for stop_info in route_detailed_stops
                    }

                    detailed_routes_data.append({
                        "vehicle_id": vehicles[vehicle_idx].id,
                        "stops": route_stops_ids, # list of location_ids
                        "segments": [], # To be populated by PathAnnotator
                        "total_distance": route_total_distance,
                        "total_time": 0, # To be calculated or estimated later
                        "capacity_utilization": 0, # To be calculated later
                        "estimated_arrival_times": estimated_arrival_times_dict
                    })

            unassigned_deliveries_ids = [
                d.id for d in deliveries if d.location_id not in all_visited_location_ids
            ]
            
            # The 'statistics' field can hold any extra info like raw arrival times if needed
            # For now, we'll put the arrival times per route directly into detailed_routes.
            return OptimizationResult(
                status='success',
                routes=processed_routes,
                total_distance=total_distance_val,
                total_cost=0.0,  # To be calculated by RouteStatsService
                assigned_vehicles=assigned_vehicles_map,
                unassigned_deliveries=unassigned_deliveries_ids,
                detailed_routes=detailed_routes_data, # Pass structured detailed routes
                statistics={} # Or add specific time window stats here
            )
        else:
            return OptimizationResult(
                status='failed',
                routes=[],
                total_distance=0.0,
                total_cost=0.0,
                assigned_vehicles={},
                unassigned_deliveries=[d.id for d in deliveries], # All deliveries unassigned
                detailed_routes=[],
                statistics={'error': 'No solution found with time window constraints!'}
            )
