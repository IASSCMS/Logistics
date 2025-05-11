import logging
import numpy as np

from typing import List, Dict, Any, Optional, Union
from route_optimizer.core.constants import MAX_SAFE_DISTANCE
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
        logger.info("Starting _add_detailed_paths method")
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
        logger.info("About to call annotator.annotate")
        # Add detailed paths using the annotator
        annotator = PathAnnotator(self.path_finder)
        annotator.annotate(result, graph)
        logger.info("Finished annotator.annotate call")
        
        # # Validate final result if it's a dict
        # if isinstance(result, dict):
        #     try:
        #         validate_optimization_result(result)
        #     except ValueError as e:
        #         logger.warning(f"Validation warning after adding paths: {e}")
        
        return result

    def _validate_inputs(self, locations, vehicles, deliveries):
        """
        Validate input data before optimization.    Args:
            locations: List of Location objects
            vehicles: List of Vehicle objects
            deliveries: List of Delivery objects    Raises:
            ValueError: If input data is invalid
        """
        # Check for empty inputs
        if not locations:
            raise ValueError("No locations provided")
        if not vehicles:
            raise ValueError("No vehicles provided")
        if not deliveries:
            raise ValueError("No deliveries provided")    # Check for valid coordinates

        for loc in locations:
            if not hasattr(loc, 'latitude') or not hasattr(loc, 'longitude'):
                raise ValueError(f"Location {loc.id} missing latitude or longitude")
            if loc.latitude < -90 or loc.latitude > 90:
                raise ValueError(f"Location {loc.id} has invalid latitude: {loc.latitude}")
            if loc.longitude < -180 or loc.longitude > 180:
                raise ValueError(f"Location {loc.id} has invalid longitude: {loc.longitude}")    # Check vehicle capacities
        # Add this to your validation function
        for loc in locations:
            if hasattr(loc, 'time_window_start') and hasattr(loc, 'time_window_end'):
                if loc.time_window_start is not None and loc.time_window_end is not None:
                    if loc.time_window_start > loc.time_window_end:
                        raise ValueError(f"Location {loc.id} has invalid time window: {loc.time_window_start} > {loc.time_window_end}")

        for vehicle in vehicles:
            if vehicle.capacity <= 0:
                raise ValueError(f"Vehicle {vehicle.id} has invalid capacity: {vehicle.capacity}")    # Check delivery demands
        # Add this to your validation function
        location_ids = {loc.id for loc in locations}
        for vehicle in vehicles:
            if vehicle.start_location_id not in location_ids:
                raise ValueError(f"Vehicle {vehicle.id} has invalid start location: {vehicle.start_location_id}")
            if vehicle.end_location_id and vehicle.end_location_id not in location_ids:
                raise ValueError(f"Vehicle {vehicle.id} has invalid end location: {vehicle.end_location_id}")

        for delivery in deliveries:
            if delivery.demand < 0:
                raise ValueError(f"Delivery {delivery.id} has negative demand: {delivery.demand}")
        # Add this to your validation function
        for delivery in deliveries:
            if delivery.location_id not in location_ids:
                raise ValueError(f"Delivery {delivery.id} has invalid location: {delivery.location_id}")

    def _convert_to_optimization_result(self, result_dict):
        """
        Convert a result dictionary to an OptimizationResult object.
        
        Args:
            result_dict: Dictionary with optimization results
            
        Returns:
            OptimizationResult object
        """
        try:
            return OptimizationResult(
                status=result_dict.get('status', 'unknown'),
                routes=result_dict.get('routes', []),
                total_distance=result_dict.get('total_distance', 0.0),
                total_cost=result_dict.get('total_cost', 0.0),
                assigned_vehicles=result_dict.get('assigned_vehicles', {}),
                unassigned_deliveries=result_dict.get('unassigned_deliveries', []),
                detailed_routes=result_dict.get('detailed_routes', []),
                statistics=result_dict.get('statistics', {})
            )
        except Exception as e:
            logger.warning(f"Failed to convert dict to OptimizationResult: {e}")
            # Return a basic result
            return OptimizationResult(
                status='error',
                routes=[],
                total_distance=0.0,
                total_cost=0.0,
                assigned_vehicles={},
                unassigned_deliveries=[],
                detailed_routes=[],
                statistics={'error': f"Conversion error: {str(e)}"}
            )

    def _sanitize_distance_matrix(self, matrix):
        """
        Sanitize distance matrix by replacing infinite or extreme values.
        
        Args:
            matrix: Distance matrix to sanitize
            
        Returns:
            Sanitized matrix
        """
        if matrix is None:
            return np.zeros((1, 1))
        
        # Make a copy to avoid modifying the original
        sanitized = np.array(matrix, dtype=float)
        
        # Define the maximum safe distance value
        max_safe_value = MAX_SAFE_DISTANCE  # This should be defined in your constants
        
        # Replace any NaN values with a large but valid distance
        sanitized = np.nan_to_num(sanitized, nan=max_safe_value)
        
        # Replace any infinite values with a large but valid distance
        sanitized[np.isinf(sanitized)] = max_safe_value
        
        # Cap any excessively large values
        sanitized[sanitized > max_safe_value] = max_safe_value
        
        # Ensure all values are non-negative
        sanitized[sanitized < 0] = 0
        
        return sanitized

    def _apply_traffic_safely(self, distance_matrix, traffic_data):
        """
        Apply traffic factors to distance matrix with bounds checking.
        
        Args:
            distance_matrix: Original distance matrix
            traffic_data: Dictionary mapping (from_idx, to_idx) to traffic factors
            
        Returns:
            Updated distance matrix
        """
        # Make a copy to avoid modifying the original
        matrix_with_traffic = np.array(distance_matrix, dtype=float)
        
        # Get matrix dimensions
        rows, cols = matrix_with_traffic.shape
        
        # Define maximum safe factor to prevent overflow
        max_safe_factor = 5.0  # Adjust this value based on your use case
        
        for (from_idx, to_idx), factor in traffic_data.items():
            # Validate indices
            if 0 <= from_idx < rows and 0 <= to_idx < cols:
                # Validate factor (ensure it's within reasonable bounds)
                safe_factor = min(max(float(factor), 1.0), max_safe_factor)
                
                # Apply the factor
                matrix_with_traffic[from_idx, to_idx] *= safe_factor
                
                # Log if factor was capped
                if safe_factor != factor:
                    logger.warning(f"Traffic factor capped from {factor} to {safe_factor} for route ({from_idx},{to_idx})")
        
        return matrix_with_traffic

    def _convert_to_optimization_result(self, result_dict):
        """
        Convert a result dictionary to an OptimizationResult object.
        
        Args:
            result_dict: Dictionary with optimization results
            
        Returns:
            OptimizationResult object
        """
        try:
            return OptimizationResult(
                status=result_dict.get('status', 'unknown'),
                routes=result_dict.get('routes', []),
                total_distance=result_dict.get('total_distance', 0.0),
                total_cost=result_dict.get('total_cost', 0.0),
                assigned_vehicles=result_dict.get('assigned_vehicles', {}),
                unassigned_deliveries=result_dict.get('unassigned_deliveries', []),
                detailed_routes=result_dict.get('detailed_routes', []),
                statistics=result_dict.get('statistics', {})
            )
        except Exception as e:
            logger.warning(f"Failed to convert dict to OptimizationResult: {e}")
            # Return a basic result
            return OptimizationResult(
                status='error',
                routes=[],
                total_distance=0.0,
                total_cost=0.0,
                assigned_vehicles={},
                unassigned_deliveries=[],
                detailed_routes=[],
                statistics={'error': f"Conversion error: {str(e)}"}
            )

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
            # Validate inputs first
            logger.info(f"Validating inputs: {len(locations)} locations, {len(vehicles)} vehicles, {len(deliveries)} deliveries")
            self._validate_inputs(locations, vehicles, deliveries)
            
            # Use provided API flag or default
            use_api_flag = use_api if use_api is not None else USE_API_BY_DEFAULT
            api_key_to_use = api_key or GOOGLE_MAPS_API_KEY
            
            logger.info(f"Creating distance matrix (use_api={use_api_flag})")
            # Create distance matrix
            distance_matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix(
                locations, use_api=use_api_flag, api_key=api_key_to_use
            )
            
            # Sanitize distance matrix before processing
            logger.debug("Sanitizing distance matrix")
            distance_matrix = self._sanitize_distance_matrix(distance_matrix)

            # Apply traffic factors if requested
            if consider_traffic and traffic_data:
                logger.info(f"Applying traffic factors to {len(traffic_data)} routes")
                # Apply traffic safely with bounds checking
                distance_matrix = self._apply_traffic_safely(distance_matrix, traffic_data)
                # Sanitize again after applying traffic
                distance_matrix = self._sanitize_distance_matrix(distance_matrix)
            
            # Find depot index
            depot_index = 0
            if locations:
                depot_service = DepotService()
                depot = depot_service.get_nearest_depot(locations)
                if depot:
                    try:
                        depot_index = location_ids.index(depot.id)
                        logger.info(f"Using depot at index {depot_index} (ID: {depot.id})")
                    except ValueError:
                        # If depot not in locations, use first location
                        logger.warning(f"Depot ID {depot.id} not found in location_ids, using first location as depot")
                        depot_index = 0
            
            # Solve the VRP
            solver = ORToolsVRPSolver()
            
            # Solve with appropriate method based on time windows
            if consider_time_windows:
                logger.info("Solving VRP with time windows")
                result = solver.solve_with_time_windows(
                    distance_matrix=distance_matrix,
                    location_ids=location_ids,
                    vehicles=vehicles,
                    deliveries=deliveries,
                    locations=locations,
                    depot_index=depot_index
                )
            else:
                logger.info("Solving VRP without time windows")
                result = solver.solve(
                    distance_matrix=distance_matrix,
                    location_ids=location_ids,
                    vehicles=vehicles,
                    deliveries=deliveries,
                    depot_index=depot_index
                )
            
            # Ensure result is a proper OptimizationResult object
            if not isinstance(result, OptimizationResult):
                logger.info("Converting result to OptimizationResult")
                result = self._convert_to_optimization_result(result)
            
            # Add detailed paths
            if result.status == 'success':
                if use_api_flag:
                    # Use Google Maps for detailed paths
                    logger.info("Adding detailed paths using Google Maps API")
                    try:
                        graph = TrafficService(api_key=api_key_to_use).create_road_graph(locations)
                        self._add_detailed_paths(result, graph, location_ids)
                    except Exception as e:
                        logger.error(f"Error adding detailed paths with Google Maps: {str(e)}")
                        # Fallback to simple paths
                        logger.info("Falling back to simple path calculation")
                        graph = {
                            'matrix': distance_matrix,
                            'location_ids': location_ids
                        }
                        PathAnnotator(self.path_finder).annotate(result, graph)
                else:
                    # Use PathAnnotator with distance matrix
                    logger.info("Adding detailed paths using distance matrix")
                    graph = {
                        'matrix': distance_matrix,
                        'location_ids': location_ids
                    }
                    PathAnnotator(self.path_finder).annotate(result, graph)
                
                # Add statistics
                logger.info("Adding route statistics")
                self._add_summary_statistics(result, vehicles)
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing routes: {str(e)}", exc_info=True)
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
