"""
Implementation of route optimization using Google OR-Tools.

This module provides classes and functions for solving Vehicle Routing Problems
(VRP) using Google's OR-Tools library.
"""
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from dataclasses import dataclass, field
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from route_optimizer.core.types_1 import Location, OptimizationResult, validate_optimization_result
from route_optimizer.models import Vehicle, Delivery

# Set up logging
logger = logging.getLogger(__name__)


# @dataclass
# class Vehicle:
#     """Class representing a vehicle with capacity and other constraints."""
#     id: str
#     capacity: float
#     start_location_id: str  # Where the vehicle starts from
#     end_location_id: Optional[str] = None  # Where the vehicle must end (if different)
#     cost_per_km: float = 1.0  # Cost per kilometer
#     fixed_cost: float = 0.0   # Fixed cost for using this vehicle
#     max_distance: Optional[float] = None  # Maximum distance the vehicle can travel
#     max_stops: Optional[int] = None  # Maximum number of stops
#     available: bool = True
#     skills: List[str] = field(default_factory=list)  # Skills/capabilities this vehicle has


# @dataclass
# class Delivery:
#     """Class representing a delivery with demand and constraints."""
#     id: str
#     location_id: str
#     demand: float  # Demand quantity
#     priority: int = 1  # 1 = normal, higher values = higher priority
#     required_skills: List[str] = field(default_factory=list)  # Required skills
#     is_pickup: bool = False  # True for pickup, False for delivery


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
        depot_index: int = 0
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
            """Returns the distance between the two nodes."""
            try:
                # Convert from routing variable Index to distance matrix NodeIndex
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                # Scale value to keep within int64 range
                # Using a smaller scale factor (100 instead of 1000) to avoid overflow
                return int(distance_matrix[from_node][to_node] * 100)
            except OverflowError:
                logger.warning(f"OverflowError in distance callback for indices: {from_index}, {to_index}")
                # Return a large but valid distance as fallback
                return 2147483647  # Maximum positive 32-bit integer
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add Capacity constraint
        def demand_callback(from_index):
            """Returns the demand for the node."""
            from_node = manager.IndexToNode(from_index)
            location_id = location_ids[from_node]
            # Find if there's a delivery at this location
            for delivery in deliveries:
                if delivery.location_id == location_id:
                    return delivery.size
            return 0
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [int(v.capacity * 100) for v in vehicles],  # Make sure the capacity is converted to integer
            True,
            'Capacity'
        )
        
        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30  # Time limit for search
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        # Process the solution
        if solution:
            routes = []
            total_distance = 0
            assigned_vehicles = {}
            
            # Extract solution routes
            for vehicle_idx in range(num_vehicles):
                route = []
                index = routing.Start(vehicle_idx)
                
                while not routing.IsEnd(index):
                    node_idx = manager.IndexToNode(index)
                    route.append(location_ids[node_idx])
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    total_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_idx
                    ) / 1000  # Convert back from int
                
                # Add the end location
                node_idx = manager.IndexToNode(index)
                route.append(location_ids[node_idx])
                
                if route:  # If the route is not empty
                    routes.append(route)
                    assigned_vehicles[vehicles[vehicle_idx].id] = len(routes) - 1
            
            # Check for unassigned deliveries
            delivery_locations = set()
            for route in routes:
                delivery_locations.update(route)
            
            unassigned_deliveries = [
                d.id for d in deliveries if d.location_id not in delivery_locations
            ]
            
            # Create the result object
            result = OptimizationResult(
                status='success',
                routes=routes,
                total_distance=total_distance,
                total_cost=0.0,  # This will be calculated later
                assigned_vehicles=assigned_vehicles,
                unassigned_deliveries=unassigned_deliveries,
                detailed_routes=[],  # Will be populated later
                statistics={}  # Will be populated later
            )
        else:
            # Solution not found
            result = OptimizationResult(
                status='failed',
                routes=[],
                total_distance=0.0,
                total_cost=0.0,
                assigned_vehicles={},
                unassigned_deliveries=[d.id for d in deliveries],
                detailed_routes=[],
                statistics={'error': 'No solution found!'}
            )
        
        # # Validate the result before returning
        # try:
        #     # Convert to dict for validation
        #     result_dict = {
        #         'status': result.status,
        #         'routes': result.routes,
        #         'total_distance': result.total_distance,
        #         'assigned_vehicles': result.assigned_vehicles,
        #         'unassigned_deliveries': result.unassigned_deliveries
        #     }
        #     validate_optimization_result(result_dict)
        # except ValueError as e:
        #     logger.error(f"Invalid optimization result: {e}")
        #     return OptimizationResult(
        #         status='failed',
        #         total_cost=0.0,
        #         statistics={'error': f"Validation error: {str(e)}"}
        #     )
        
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
    ) -> Dict[str, Any]:
        """
        Solve the Vehicle Routing Problem with Time Windows.

        Returns:
            Dictionary containing the solution details with route time information.
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
                return {'status': 'failed', 'error': f"Vehicle location not found: {e}"}

        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            try:
                # Convert from routing variable Index to distance matrix NodeIndex
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                # Scale value to keep within int64 range
                # Using a smaller scale factor (100 instead of 1000) to avoid overflow
                return int(distance_matrix[from_node][to_node] * 100)
            except OverflowError:
                logger.warning(f"OverflowError in distance callback for indices: {from_index}, {to_index}")
                # Return a large but valid distance as fallback
                return 2147483647  # Maximum positive 32-bit integer

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Time callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            distance_km = distance_matrix[from_node][to_node]
            travel_minutes = (distance_km / speed_km_per_hour) * 60
            to_loc = location_index_to_location.get(to_node)
            service_time = to_loc.service_time if to_loc else 0
            return int((travel_minutes + service_time) * 60)  # seconds

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
        for idx, location_id in enumerate(location_ids):
            loc = location_index_to_location.get(idx)
            if loc and loc.time_window_start is not None and loc.time_window_end is not None:
                start = loc.time_window_start * 60
                end = loc.time_window_end * 60
                index = manager.NodeToIndex(idx)
                time_dimension.CumulVar(index).SetRange(start, end)

        # Add capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location_id = location_ids[from_node]
            total_demand = 0
            for d in deliveries:
                if d.location_id == location_id:
                    total_demand += -d.demand if d.is_pickup else d.demand
            return int(total_demand * 100)

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [int(v.capacity * 100) for v in vehicles],
            True,
            'Capacity'
        )

        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = self.time_limit_seconds

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            routes = []
            assigned_vehicles = {}
            total_distance = 0
            delivery_locations = set()

            for vehicle_idx in range(num_vehicles):
                route = []
                index = routing.Start(vehicle_idx)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    time_var = time_dimension.CumulVar(index)
                    time_val = solution.Min(time_var)
                    route.append({
                        'location_id': location_ids[node_index],
                        'arrival_time_seconds': time_val
                    })
                    delivery_locations.add(location_ids[node_index])
                    prev_index = index
                    index = solution.Value(routing.NextVar(index))
                    total_distance += routing.GetArcCostForVehicle(prev_index, index, vehicle_idx) / 1000
                node_index = manager.IndexToNode(index)
                time_val = solution.Min(time_dimension.CumulVar(index))
                route.append({
                    'location_id': location_ids[node_index],
                    'arrival_time_seconds': time_val
                })
                if route:
                    routes.append(route)
                    assigned_vehicles[vehicles[vehicle_idx].id] = len(routes) - 1

            unassigned_deliveries = [
                d.id for d in deliveries if d.location_id not in delivery_locations
            ]

            return {
                'status': 'success',
                'routes': routes,
                'total_distance': total_distance,
                'assigned_vehicles': assigned_vehicles,
                'unassigned_deliveries': unassigned_deliveries
            }
        else:
            return {
                'status': 'failed',
                'error': 'No solution found with time window constraints!'
            }
