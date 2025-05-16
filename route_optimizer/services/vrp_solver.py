from typing import Dict, Any
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from route_optimizer.models.vrp_input import VRPInput


def solve_cvrp(vrp_input: VRPInput) -> Dict[str, Any]:
    """
    Solves a Capacitated Vehicle Routing Problem (CVRP) using OR-Tools.

    Args:
        vrp_input (VRPInput): Prepared VRP input object.

    Returns:
        dict: {
            'status': 'success' or 'failed',
            'routes': list of routes (each a list of node indices),
            'total_distance': total distance across all routes
        }
    """
    vrp_input.validate()

    manager = pywrapcp.RoutingIndexManager(
        len(vrp_input.distance_matrix),
        vrp_input.num_vehicles,
        vrp_input.starts,
        vrp_input.ends
    )

    routing = pywrapcp.RoutingModel(manager)

    # Define distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return vrp_input.distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )

    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    for request in vrp_input.pickups_deliveries:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        # The same vehicle should do pickup and delivery
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
        )
        # Should pick up before deliver
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index)
            <= distance_dimension.CumulVar(delivery_index)
        )

    # Define demand callback
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return vrp_input.demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        vrp_input.vehicle_capacities,
        True,
        "Capacity"
    )

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    search_parameters.solution_limit = 1

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return {"status": "failed", "error": "No solution found."}

    # Extract routes
    routes = []
    total_distance = 0
    for vehicle_id in range(vrp_input.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            next_index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(index, next_index, vehicle_id)
            index = next_index
        route.append(manager.IndexToNode(index))
        if len(route) > 2:
            routes.append(route)

    return {
        "status": "success",
        "routes": routes,
        "total_distance": total_distance,
    }
