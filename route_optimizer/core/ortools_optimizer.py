from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
from dataclasses import dataclass, field
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from route_optimizer.core.distance_matrix import Location

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Vehicle:
    id: str
    capacity: float
    start_location_id: str
    end_location_id: Optional[str] = None
    cost_per_km: float = 1.0
    fixed_cost: float = 0.0
    max_distance: Optional[float] = None
    max_stops: Optional[int] = None
    available: bool = True
    skills: List[str] = field(default_factory=list)

@dataclass
class Delivery:
    id: str
    location_id: str
    demand: float
    priority: int = 1
    required_skills: List[str] = field(default_factory=list)
    is_pickup: bool = False

class ORToolsVRPSolver:
    def __init__(self, time_limit_seconds: int = 30):
        self.time_limit_seconds = time_limit_seconds

    def solve(
        self,
        distance_matrix: np.ndarray,
        location_ids: List[str],
        vehicles: List[Vehicle],
        deliveries: List[Delivery],
        depot_index: int = 0
    ) -> Dict[str, Any]:
        num_locations = len(location_ids)
        num_vehicles = len(vehicles)
        location_id_to_index = {loc_id: idx for idx, loc_id in enumerate(location_ids)}
        starts, ends = [], []

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

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location_id = location_ids[from_node]
            total_demand = sum(
                (-d.demand if d.is_pickup else d.demand)
                for d in deliveries if d.location_id == location_id
            )
            return int(total_demand * 100)

        solution = self.get_solution(demand_callback, routing, vehicles)

        if solution:
            routes, assigned_vehicles, total_distance = [], {}, 0
            for vehicle_idx in range(num_vehicles):
                route, index = [], routing.Start(vehicle_idx)
                while not routing.IsEnd(index):
                    node_idx = manager.IndexToNode(index)
                    route.append(location_ids[node_idx])
                    previous_index, index = index, solution.Value(routing.NextVar(index))
                    total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx) / 1000
                node_idx = manager.IndexToNode(index)
                route.append(location_ids[node_idx])
                if route:
                    routes.append(route)
                    assigned_vehicles[vehicles[vehicle_idx].id] = len(routes) - 1

            delivery_locations = {loc_id for route in routes for loc_id in route}
            unassigned = [d.id for d in deliveries if d.location_id not in delivery_locations]

            return {
                'status': 'success',
                'routes': routes,
                'total_distance': total_distance,
                'assigned_vehicles': assigned_vehicles,
                'unassigned_deliveries': unassigned
            }
        else:
            return {'status': 'failed', 'error': 'No solution found!'}

    def solve_with_time_windows(
        self,
        distance_matrix: np.ndarray,
        location_ids: List[str],
        vehicles: List[Vehicle],
        deliveries: List[Delivery],
        locations: List[Location],
        pickup_delivery_pairs: Optional[List[Tuple[str, str]]] = None,
        depot_index: int = 0,
        speed_km_per_hour: float = 50.0
    ) -> Dict[str, Any]:
        pickup_delivery_pairs = pickup_delivery_pairs or []
        num_locations = len(location_ids)
        num_vehicles = len(vehicles)
        location_id_to_index = {loc_id: idx for idx, loc_id in enumerate(location_ids)}
        location_index_to_location = {
            idx: next((loc for loc in locations if loc.id == loc_id), None)
            for idx, loc_id in enumerate(location_ids)
        }

        starts, ends = [], []
        for vehicle in vehicles:
            try:
                start_idx = location_id_to_index[vehicle.start_location_id]
                end_idx = location_id_to_index.get(vehicle.end_location_id or vehicle.start_location_id, start_idx)
                starts.append(start_idx)
                ends.append(end_idx)
            except KeyError as e:
                logger.error(f"Vehicle location not found: {e}")
                return {'status': 'failed', 'error': f"Vehicle location not found: {e}"}

        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def time_callback(from_index, to_index):
            from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            distance_km = distance_matrix[from_node][to_node]
            travel_minutes = (distance_km / speed_km_per_hour) * 60
            service_time = location_index_to_location.get(to_node).service_time if location_index_to_location.get(to_node) else 0
            return int((travel_minutes + service_time) * 60)

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(time_callback_index, 3600, 86400, False, 'Time')
        time_dimension = routing.GetDimensionOrDie('Time')

        for idx, location_id in enumerate(location_ids):
            loc = location_index_to_location.get(idx)
            if loc and loc.time_window_start is not None and loc.time_window_end is not None:
                time_dimension.CumulVar(manager.NodeToIndex(idx)).SetRange(loc.time_window_start * 60, loc.time_window_end * 60)

        for pickup_id, delivery_id in pickup_delivery_pairs:
            if pickup_id in location_id_to_index and delivery_id in location_id_to_index:
                pickup_index = manager.NodeToIndex(location_id_to_index[pickup_id])
                delivery_index = manager.NodeToIndex(location_id_to_index[delivery_id])
                routing.AddPickupAndDelivery(pickup_index, delivery_index)
                routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
                routing.solver().Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location_id = location_ids[from_node]
            total_demand = sum(-d.demand if d.is_pickup else d.demand for d in deliveries if d.location_id == location_id)
            return int(total_demand * 100)

        solution = self.get_solution(demand_callback, routing, vehicles)

        if solution:
            routes, assigned_vehicles, total_distance, delivery_locations = [], {}, 0, set()
            for vehicle_idx in range(num_vehicles):
                route, index = [], routing.Start(vehicle_idx)
                while not routing.IsEnd(index):
                    node_idx = manager.IndexToNode(index)
                    time_val = solution.Min(time_dimension.CumulVar(index))
                    route.append({'location_id': location_ids[node_idx], 'arrival_time_seconds': time_val})
                    delivery_locations.add(location_ids[node_idx])
                    prev_index = index
                    index = solution.Value(routing.NextVar(index))
                    total_distance += routing.GetArcCostForVehicle(prev_index, index, vehicle_idx) / 1000
                node_idx = manager.IndexToNode(index)
                time_val = solution.Min(time_dimension.CumulVar(index))
                route.append({'location_id': location_ids[node_idx], 'arrival_time_seconds': time_val})
                if route:
                    routes.append(route)
                    assigned_vehicles[vehicles[vehicle_idx].id] = len(routes) - 1

            unassigned = [d.id for d in deliveries if d.location_id not in delivery_locations]

            return {
                'status': 'success',
                'routes': routes,
                'total_distance': total_distance,
                'assigned_vehicles': assigned_vehicles,
                'unassigned_deliveries': unassigned
            }
        return {'status': 'failed', 'error': 'No solution found with time window constraints!'}

    def get_solution(self, demand_callback, routing, vehicles):
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [int(v.capacity * 100) for v in vehicles],
                                                True, 'Capacity')
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = self.time_limit_seconds
        solution = routing.SolveWithParameters(search_parameters)
        return solution
