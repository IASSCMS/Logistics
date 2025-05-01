import logging
from route_optimizer.services.traffic_service import TrafficService
from route_optimizer.services.depot_service import DepotService
from route_optimizer.services.path_annotation_service import PathAnnotator
from route_optimizer.services.route_stats_service import RouteStatsService
from route_optimizer.core.dijkstra import DijkstraPathFinder
from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver
from route_optimizer.core.distance_matrix import DistanceMatrixBuilder

logger = logging.getLogger(__name__)

class OptimizationService:
    def __init__(self, time_limit_seconds=30):
        self.vrp_solver = ORToolsVRPSolver(time_limit_seconds)
        self.path_finder = DijkstraPathFinder()

    def optimize_routes(self, locations, vehicles, deliveries, consider_traffic=False, consider_time_windows=False, traffic_data=None):
        distance_matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix(locations, use_haversine=True)

        if consider_traffic and traffic_data:
            distance_matrix = TrafficService.apply_traffic_factors(distance_matrix, traffic_data)

        depot_index = DepotService.find_depot_index(locations)

        if consider_time_windows:
            result = self.vrp_solver.solve_with_time_windows(
                distance_matrix=distance_matrix,
                location_ids=location_ids,
                vehicles=vehicles,
                deliveries=deliveries,
                locations=locations,
                depot_index=depot_index
            )
        else:
            result = self.vrp_solver.solve(
                distance_matrix=distance_matrix,
                location_ids=location_ids,
                vehicles=vehicles,
                deliveries=deliveries,
                depot_index=depot_index
            )

        if result['status'] == 'success':
            graph = DistanceMatrixBuilder.distance_matrix_to_graph(distance_matrix, location_ids)
            annotator = PathAnnotator(self.path_finder)
            annotator.annotate(result, graph)
            RouteStatsService.add_statistics(result, vehicles)

        return result
