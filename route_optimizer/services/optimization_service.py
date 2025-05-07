import logging

from typing import List, Dict, Any, Optional
from route_optimizer.services.path_annotation_service import PathAnnotator
from route_optimizer.core.dijkstra import DijkstraPathFinder
from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver
from route_optimizer.settings import USE_API_BY_DEFAULT, GOOGLE_MAPS_API_KEY
from route_optimizer.models import Location, Vehicle, Delivery
from route_optimizer.core.distance_matrix import DistanceMatrixBuilder
from route_optimizer.services.depot_service import DepotService
from route_optimizer.services.traffic_service import TrafficService
from route_optimizer.services.route_stats_service import RouteStatsService

logger = logging.getLogger(__name__)

class OptimizationService:
    def __init__(self, vrp_solver, path_finder):
        self.vrp_solver = vrp_solver
        self.path_finder = path_finder
        
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
    ) -> Dict[str, Any]:
        """
        Optimize vehicle routes using OR-Tools.
        
        Args:
            locations: List of Location objects.
            vehicles: List of Vehicle objects.
            deliveries: List of Delivery objects.
            consider_traffic: Whether to apply traffic factors.
            consider_time_windows: Whether to consider time windows.
            traffic_data: Dictionary of traffic factors.
            use_api: Whether to use Google Distance Matrix API.
            api_key: Google API key.
            
        Returns:
            Dictionary with optimization results.
        """
        # If use_api is not specified, use the default setting
        if use_api is None:
            use_api = USE_API_BY_DEFAULT
            
        api_key = api_key or GOOGLE_MAPS_API_KEY
            
        try:
            # Try to use the Google Distance Matrix API if requested
            if use_api and api_key:
                logger.info("Using Google Distance Matrix API for route optimization")
                distance_matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix_from_api(
                    locations, api_key
                )
            else:
                logger.info("Using Haversine distance calculation for route optimization")
                distance_matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix(
                    locations, use_haversine=True
                )

            if consider_traffic and traffic_data:
                logger.info("Applying traffic factors to distance matrix")
                distance_matrix = TrafficService.apply_traffic_factors(distance_matrix, traffic_data)

            depot_index = DepotService.find_depot_index(locations)

            if consider_time_windows:
                logger.info("Solving VRP with time windows")
                result = self.vrp_solver.solve_with_time_windows(
                    distance_matrix=distance_matrix,
                    location_ids=location_ids,
                    vehicles=vehicles,
                    deliveries=deliveries,
                    locations=locations,
                    depot_index=depot_index
                )
            else:
                logger.info("Solving VRP without time windows")
                result = self.vrp_solver.solve(
                    distance_matrix=distance_matrix,
                    location_ids=location_ids,
                    vehicles=vehicles,
                    deliveries=deliveries,
                    depot_index=depot_index
                )

            if result.get('status') == 'success':
                logger.info("VRP solved successfully, annotating results")
                graph = DistanceMatrixBuilder.distance_matrix_to_graph(distance_matrix, location_ids)
                annotator = PathAnnotator(self.path_finder)
                annotator.annotate(result, graph)
                RouteStatsService.add_statistics(result, vehicles)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in route optimization: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Route optimization failed: {str(e)}",
                'routes': [],
                'unassigned_deliveries': [delivery.id for delivery in deliveries]
            }

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
