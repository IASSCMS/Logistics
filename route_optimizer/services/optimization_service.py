import logging

from typing import List, Dict, Any, Optional, Union
from route_optimizer.services.path_annotation_service import PathAnnotator
from route_optimizer.core.dijkstra import DijkstraPathFinder
from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver
from route_optimizer.settings import USE_API_BY_DEFAULT, GOOGLE_MAPS_API_KEY
from route_optimizer.core.types_1 import DetailedRoute, Location, OptimizationResult, validate_optimization_result
from route_optimizer.models import Vehicle, Delivery
from route_optimizer.core.distance_matrix import DistanceMatrixBuilder
from route_optimizer.services.depot_service import DepotService
from route_optimizer.services.traffic_service import TrafficService
from route_optimizer.services.route_stats_service import RouteStatsService

logger = logging.getLogger(__name__)

class OptimizationService:
    def __init__(self, vrp_solver=None, path_finder=None):
        """
        Initialize the optimization service.
        
        Args:
            vrp_solver: The VRP solver to use. If None, a default ORToolsVRPSolver will be created.
            path_finder: The path finder to use. If None, a default DijkstraPathFinder will be created.
        """
        from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver
        from route_optimizer.core.dijkstra import DijkstraPathFinder
        self.vrp_solver = vrp_solver or ORToolsVRPSolver()
        self.path_finder = path_finder or DijkstraPathFinder()

    def _create_pathfinder(self):
        """
        Create a path finder instance.
        
        Returns:
            A path finder instance
        """
        from route_optimizer.core.dijkstra import DijkstraPathFinder
        return DijkstraPathFinder()

    def _add_summary_statistics(self, result, vehicles):
        """
        Add summary statistics to the optimization result.
        
        Args:
            result: The optimization result to enrich
            vehicles: List of vehicles used in optimization
            
        Returns:
            The enriched result with summary statistics
        """
        from route_optimizer.services.route_stats_service import RouteStatsService
        RouteStatsService.add_statistics(result, vehicles)
        return result
    
    def _add_detailed_paths(self, result, graph, location_ids=None):
        """
        Add detailed path information to the optimization result.
        
        Args:
            result: The optimization result to enrich
            graph: The graph representation of the distance matrix
            location_ids: Optional list of location IDs
        """
        # Handle both Dict and OptimizationResult types
        if isinstance(result, OptimizationResult):
            # Working with DTO
            if not result.detailed_routes:
                result.detailed_routes = []
                
            # Add routes if available and detailed_routes is empty
            if result.routes and not result.detailed_routes:
                for route_idx, route in enumerate(result.routes):
                    # Find which vehicle is assigned to this route
                    vehicle_id = None
                    if result.assigned_vehicles:
                        for v_id, v_route_idx in result.assigned_vehicles.items():
                            if v_route_idx == route_idx:
                                vehicle_id = v_id
                                break
                    
                    # Convert indices to IDs if needed
                    if location_ids and all(isinstance(stop, int) for stop in route):
                        stops = [location_ids[stop] for stop in route]
                    else:
                        stops = route
                    
                    # Create detailed route using DTO
                    detailed_route = DetailedRoute(
                        vehicle_id=vehicle_id or f"unknown_{route_idx}",
                        stops=stops,
                        segments=[]
                    )
                    result.detailed_routes.append(vars(detailed_route))
            
            # Ensure each detailed route has a vehicle_id
            for route_idx, route in enumerate(result.detailed_routes):
                if 'vehicle_id' not in route and result.assigned_vehicles:
                    for v_id, v_route_idx in result.assigned_vehicles.items():
                        if v_route_idx == route_idx:
                            route['vehicle_id'] = v_id
                            break
        else:
            # Working with dict (backward compatibility)
            # Create detailed_routes if not present
            if 'detailed_routes' not in result:
                result['detailed_routes'] = []
                
            # Add routes if available and detailed_routes is empty
            if 'routes' in result and not result['detailed_routes']:
                for route_idx, route in enumerate(result['routes']):
                    # Find which vehicle is assigned to this route
                    vehicle_id = None
                    if 'assigned_vehicles' in result:
                        for v_id, v_route_idx in result['assigned_vehicles'].items():
                            if v_route_idx == route_idx:
                                vehicle_id = v_id
                                break
                    
                    # Convert indices to IDs if needed
                    if location_ids and all(isinstance(stop, int) for stop in route):
                        stops = [location_ids[stop] for stop in route]
                    else:
                        stops = route
                    
                    # Create detailed route
                    detailed_route = {
                        'stops': stops,
                        'segments': [],
                        'vehicle_id': vehicle_id or f"unknown_{route_idx}"  # Ensure vehicle_id is added with fallback
                    }
                    result['detailed_routes'].append(detailed_route)
            
            # Ensure each detailed route has a vehicle_id
            for route_idx, route in enumerate(result['detailed_routes']):
                if 'vehicle_id' not in route and 'assigned_vehicles' in result:
                    for v_id, v_route_idx in result['assigned_vehicles'].items():
                        if v_route_idx == route_idx:
                            route['vehicle_id'] = v_id
                            break
                
                # Add default vehicle_id if still missing
                if 'vehicle_id' not in route:
                    route['vehicle_id'] = f"unknown_{route_idx}"
        
        # Add detailed paths using the annotator
        annotator = PathAnnotator(self.path_finder)
        annotator.annotate(result, graph)
        
        # # Validate final result if it's a dict
        # if isinstance(result, dict):
        #     try:
        #         validate_optimization_result(result)
        #     except ValueError as e:
        #         logger.warning(f"Validation warning after adding paths: {e}")
        
        return result

    def optimize_routes(
        self,
        locations: List[Location],
        vehicles: List[Vehicle],
        deliveries: List[Delivery],
        consider_traffic: bool = False,
        consider_time_windows: bool = False,
        traffic_data: Optional[Dict[str, Any]] = None,
        use_api: Optional[bool] = None,
        api_key: Optional[str] = None
    ) -> OptimizationResult:
        """
        Optimize vehicle routes using OR-Tools.

        Args:
            locations: List of Location objects with coordinates and other attributes.
            vehicles: List of Vehicle objects with capacity and other constraints.
            deliveries: List of Delivery objects with demands and constraints.
            consider_traffic: Whether to apply traffic factors to travel times.
            consider_time_windows: Whether to consider time windows in the optimization.
            traffic_data: Optional dictionary mapping location pairs to traffic factors.
            use_api: Whether to use external APIs for distance calculations.
            api_key: API key for external services if applicable.

        Returns:
            OptimizationResult object containing:
                - status: Success/failure status of the optimization
                - routes: List of routes, each containing a list of location IDs
                - total_distance: Total distance of all routes
                - total_cost: Total cost of all routes
                - assigned_vehicles: Dictionary mapping vehicle IDs to route indices
                - unassigned_deliveries: List of delivery IDs that couldn't be assigned
                - detailed_routes: List of detailed route information including segments
                - statistics: Dictionary of statistics about the optimization
        """
        try:
            # Use provided API flag or default
            use_api_flag = use_api if use_api is not None else USE_API_BY_DEFAULT
            api_key_to_use = api_key or GOOGLE_MAPS_API_KEY
            
            # Create distance matrix
            distance_matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix(
                locations, use_api=use_api_flag, api_key=api_key_to_use
            )
            
            # Apply traffic factors if requested
            if consider_traffic and traffic_data:
                # Apply traffic data to distance matrix
                for (from_idx, to_idx), factor in traffic_data.items():
                    if 0 <= from_idx < len(distance_matrix) and 0 <= to_idx < len(distance_matrix[0]):
                        distance_matrix[from_idx][to_idx] *= factor
            
            # Find depot index
            depot_index = 0
            if locations:
                depot_service = DepotService()
                depot = depot_service.get_nearest_depot(locations)
                if depot:
                    try:
                        depot_index = location_ids.index(depot.id)
                    except ValueError:
                        # If depot not in locations, use first location
                        depot_index = 0
            
            # Solve the VRP
            solver = ORToolsVRPSolver()
            result = solver.solve(
                distance_matrix=distance_matrix,
                location_ids=location_ids,
                vehicles=vehicles,
                deliveries=deliveries,
                depot_index=depot_index
            )
            
            # # Ensure result is properly structured
            # if isinstance(result, dict) and not isinstance(result, OptimizationResult):
            #     # If result is still a dict, convert to OptimizationResult
            #     try:
            #         result = OptimizationResult(
            #             status=result.get('status', 'unknown'),
            #             routes=result.get('routes', []),
            #             total_distance=result.get('total_distance', 0.0),
            #             total_cost=result.get('total_cost', 0.0),
            #             assigned_vehicles=result.get('assigned_vehicles', {}),
            #             unassigned_deliveries=result.get('unassigned_deliveries', []),
            #             detailed_routes=result.get('detailed_routes', []),
            #             statistics=result.get('statistics', {})
            #         )
            #     except Exception as e:
            #         logger.warning(f"Failed to convert dict to OptimizationResult: {e}")
            
            # # Validate the result
            # if isinstance(result, OptimizationResult):
            #     validate_optimization_result(result)
            
            # # Add statistics if they don't exist
            # if isinstance(result, OptimizationResult) and not result.statistics:
            #     result.statistics = {}
            #     self.route_stats_service.add_statistics(result, vehicles)
            # elif isinstance(result, dict) and 'statistics' not in result:
            #     result['statistics'] = {}
            #     self.route_stats_service.add_statistics(result, vehicles)
            
            # Add detailed paths
            if use_api_flag:
                # Use Google Maps for detailed paths
                graph = TrafficService(api_key=api_key_to_use).create_road_graph(locations)
                self.add_detailed_paths(result, graph, location_ids)
            else:
                # Use PathAnnotator with distance matrix
                graph = {
                    'matrix': distance_matrix,
                    'location_ids': location_ids
                }
                # PathAnnotator().annotate(result, graph)
                PathAnnotator(self.path_finder).annotate(result, graph)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in route optimization: {str(e)}", exc_info=True)
            # Return an error result as OptimizationResult
            return OptimizationResult(
                status='error',
                routes=[],
                total_distance=0.0,
                total_cost=0.0,
                assigned_vehicles={},
                unassigned_deliveries=[delivery.id for delivery in deliveries],
                detailed_routes=[],
                statistics={'error': str(e)}
            )

# class OptimizationService:
    # def __init__(self, time_limit_seconds=30):
    #     self.vrp_solver = ORToolsVRPSolver(time_limit_seconds)
    #     self.path_finder = DijkstraPathFinder()

    # def optimize_routes(self, locations, vehicles, deliveries, consider_traffic=False, consider_time_windows=False, traffic_data=None):
    #     distance_matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix(locations, use_haversine=True)

    #     if consider_traffic and traffic_data:
    #         distance_matrix = TrafficService.apply_traffic_factors(distance_matrix, traffic_data)

    #     depot_index = DepotService.find_depot_index(locations)

    #     if consider_time_windows:
    #         result = self.vrp_solver.solve_with_time_windows(
    #             distance_matrix=distance_matrix,
    #             location_ids=location_ids,
    #             vehicles=vehicles,
    #             deliveries=deliveries,
    #             locations=locations,
    #             depot_index=depot_index
    #         )
    #     else:
    #         result = self.vrp_solver.solve(
    #             distance_matrix=distance_matrix,
    #             location_ids=location_ids,
    #             vehicles=vehicles,
    #             deliveries=deliveries,
    #             depot_index=depot_index
    #         )

    #     if result['status'] == 'success':
    #         graph = DistanceMatrixBuilder.distance_matrix_to_graph(distance_matrix, location_ids)
    #         annotator = PathAnnotator(self.path_finder)
    #         annotator.annotate(result, graph)
    #         RouteStatsService.add_statistics(result, vehicles)

    #     return result
