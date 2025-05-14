import logging
import numpy as np
import hashlib
import json
import dataclasses # For converting DTO to dict

from typing import List, Dict, Any, Optional, Union, Tuple
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
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)

class OptimizationService:
    def __init__(self, time_limit_seconds=30, vrp_solver=None, path_finder=None):
        """
        Initialize the optimization service.
        
        Args:
            vrp_solver: The VRP solver to use. If None, a default ORToolsVRPSolver will be created.
            path_finder: The path finder to use. If None, a default DijkstraPathFinder will be created.
        """
        from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver
        from route_optimizer.core.dijkstra import DijkstraPathFinder
        self.vrp_solver = vrp_solver or ORToolsVRPSolver(time_limit_seconds)
        self.path_finder = path_finder or DijkstraPathFinder()

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
    
    def _add_detailed_paths(self, result, graph, location_ids=None, annotator=None):
        """
        Add detailed path information to the optimization result.
        
        Args:
            result: The optimization result to enrich
            graph: The graph representation of the distance matrix
            location_ids: Optional list of location IDs
        """
        logger.info("Starting _add_detailed_paths method")
        
        # Store original total_distance based on result type
        is_dto = isinstance(result, OptimizationResult)
        original_total_distance = result.total_distance if is_dto else result.get('total_distance')
        
        # Handle both Dict and OptimizationResult types
        if is_dto:
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
                            
                # Add default vehicle_id if still missing
                if 'vehicle_id' not in route:
                    route['vehicle_id'] = f"unknown_{route_idx}"
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
        
        # Restore original total_distance appropriately based on type
        if is_dto:
            result.total_distance = original_total_distance
        elif original_total_distance is not None:
            result['total_distance'] = original_total_distance
        
        # Add detailed paths using the annotator
        if annotator is None:
            annotator = PathAnnotator(self.path_finder)
        logger.info("About to call annotator.annotate")
        annotator.annotate(result, graph)
        logger.info("Finished annotator.annotate call")
        
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
            try:
                if not hasattr(loc, 'latitude') or not hasattr(loc, 'longitude'):
                    raise ValueError(f"Location {loc.id} missing latitude or longitude")
                
                # First check if latitude or longitude are None
                if loc.latitude is None or loc.longitude is None:
                    raise ValueError(f"Location {loc.id} is missing latitude or longitude coordinates")
                
                # Only validate ranges if they're not None
                if loc.latitude < -90 or loc.latitude > 90:
                    raise ValueError(f"Location {loc.id} has invalid latitude: {loc.latitude}")
                if loc.longitude < -180 or loc.longitude > 180:
                    raise ValueError(f"Location {loc.id} has invalid longitude: {loc.longitude}")
            except AttributeError:
                raise ValueError(f"Location {loc.id} has invalid coordinate attributes")
                
        # Check time windows
        for loc in locations:
            if hasattr(loc, 'time_window_start') and hasattr(loc, 'time_window_end'):
                if loc.time_window_start is not None and loc.time_window_end is not None:
                    if loc.time_window_start > loc.time_window_end:
                        raise ValueError(f"Location {loc.id} has invalid time window: {loc.time_window_start} > {loc.time_window_end}")

        # Check vehicle capacities
        for vehicle in vehicles:
            if vehicle.capacity <= 0:
                raise ValueError(f"Vehicle {vehicle.id} has invalid capacity: {vehicle.capacity}")
                
        # Check location references
        location_ids = {loc.id for loc in locations}
        for vehicle in vehicles:
            if vehicle.start_location_id not in location_ids:
                raise ValueError(f"Vehicle {vehicle.id} has invalid start location: {vehicle.start_location_id}")
            if vehicle.end_location_id and vehicle.end_location_id not in location_ids:
                raise ValueError(f"Vehicle {vehicle.id} has invalid end location: {vehicle.end_location_id}")

        # Check delivery demands and locations
        for delivery in deliveries:
            if delivery.demand < 0:
                raise ValueError(f"Delivery {delivery.id} has negative demand: {delivery.demand}")
            if delivery.location_id not in location_ids:
                raise ValueError(f"Delivery {delivery.id} has invalid location: {delivery.location_id}")

    def _generate_cache_key(self, locations: List[Location], vehicles: List[Vehicle], 
                            deliveries: List[Delivery], consider_traffic: bool, 
                            consider_time_windows: bool, 
                            traffic_data: Optional[Dict[Tuple[int, int], float]], 
                            use_api: Optional[bool], api_key: Optional[str]) -> str:
        """Generates a deterministic cache key from the input parameters."""
        # Convert DTOs to dicts for stable serialization
        # Ensure consistent ordering for lists of dicts by sorting based on a unique key (e.g., id)
        # For tuples in traffic_data keys, convert them to strings
        
        key_parts = {
            "locations": sorted([dataclasses.asdict(loc) for loc in locations], key=lambda x: x['id']),
            "vehicles": sorted([dataclasses.asdict(veh) for veh in vehicles], key=lambda x: x['id']),
            "deliveries": sorted([dataclasses.asdict(deliv) for deliv in deliveries], key=lambda x: x['id']),
            "consider_traffic": consider_traffic,
            "consider_time_windows": consider_time_windows,
            "use_api": use_api if use_api is not None else settings.USE_API_BY_DEFAULT, # Use effective value
            # "api_key": api_key, # Avoid caching based on api_key if it changes frequently but data doesn't
                                # Or include if results are truly different per key for same data
        }
        # Convert traffic_data keys (tuples) to strings for JSON serialization
        if traffic_data:
            key_parts["traffic_data"] = {f"{k[0]}-{k[1]}": v for k, v in sorted(traffic_data.items())}
        
        # Serialize to a JSON string with sorted keys for deterministic output
        serialized_params = json.dumps(key_parts, sort_keys=True)
        
        # Hash the string to create a manageable key
        return "opt_result_" + hashlib.md5(serialized_params.encode('utf-8')).hexdigest()
    
    def optimize_routes(
        self,
        locations: List[Location],
        vehicles: List[Vehicle],
        deliveries: List[Delivery],
        consider_traffic: bool = False,
        consider_time_windows: bool = False,
        traffic_data: Optional[Dict[Tuple[int, int], float]] = None,
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
        # --- Caching Logic ---
        cache_key = self._generate_cache_key(
            locations, vehicles, deliveries, consider_traffic, 
            consider_time_windows, traffic_data, use_api, api_key
        )
        cached_result_dict = cache.get(cache_key)
        
        if cached_result_dict:
            logger.info(f"Returning cached OptimizationResult for key: {cache_key}")
            return OptimizationResult.from_dict(cached_result_dict) # Reconstruct DTO from cached dict
        
        logger.info(f"No cache hit for key: {cache_key}. Proceeding with optimization.")
        # --- End Caching Logic ---
        
        try:
            # Validate inputs first
            logger.info(f"Validating inputs: {len(locations)} locations, {len(vehicles)} vehicles, {len(deliveries)} deliveries")
            self._validate_inputs(locations, vehicles, deliveries)
            
            # Use provided API flag or default
            use_api_flag = use_api if use_api is not None else USE_API_BY_DEFAULT
            api_key_to_use = api_key or GOOGLE_MAPS_API_KEY
            
            # In optimize_routes method, replace the distance matrix creation with:
            if use_api_flag:
                # When using API, only pass the required parameters the test expects
                logger.info(f"Creating distance matrix using API")
                distance_matrix, _, location_ids = DistanceMatrixBuilder.create_distance_matrix(
                    locations, 
                    use_api=use_api_flag,
                    api_key=api_key_to_use
                )
            else:
                # When not using API, use the existing parameters
                logger.info(f"Creating distance matrix using Haversine calculation")
                distance_matrix, _, location_ids = DistanceMatrixBuilder.create_distance_matrix(
                    locations, 
                    use_haversine=True, 
                    distance_calculation="haversine",
                    use_api=False,
                    api_key=None
                )
            
            # Sanitize distance matrix before processing
            logger.debug("Sanitizing distance matrix")
            # Call the static method from DistanceMatrixBuilder
            distance_matrix = DistanceMatrixBuilder._sanitize_distance_matrix(distance_matrix)

            # Apply traffic factors if requested
            if consider_traffic and traffic_data:
                logger.info(f"Applying traffic factors to {len(traffic_data)} routes")
                # Apply traffic safely with bounds checking
                distance_matrix = DistanceMatrixBuilder.add_traffic_factors(distance_matrix, traffic_data)
                # Sanitize again after applying traffic
                distance_matrix = DistanceMatrixBuilder._sanitize_distance_matrix(distance_matrix)
            
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
            
            # Solve with appropriate method based on time windows
            if consider_time_windows:
                logger.info("Solving VRP with time windows")
                raw_solver_result: OptimizationResult = self.vrp_solver.solve_with_time_windows(
                    distance_matrix=distance_matrix,
                    location_ids=location_ids,
                    vehicles=vehicles,
                    deliveries=deliveries,
                    locations=locations,
                    depot_index=depot_index
                )
            else:
                logger.info("Solving VRP without time windows")
                raw_solver_result: OptimizationResult = self.vrp_solver.solve(
                    distance_matrix=distance_matrix,
                    location_ids=location_ids,
                    vehicles=vehicles,
                    deliveries=deliveries,
                    depot_index=depot_index
                )
            
            result = raw_solver_result
            
            # Defensive check: If the solver, unexpectedly, did not return an OptimizationResult DTO.
            # Given that your ORToolsVRPSolver methods are typed to return OptimizationResult,
            # this block might rarely execute but provides robustness.
            if not isinstance(result, OptimizationResult):
                logger.warning(
                    "Solver did not return an OptimizationResult DTO as expected. "
                    f"Got {type(result)}. Attempting conversion from dict."
                )
                # Assuming raw_solver_result (now 'result') was a dict if not a DTO
                result = OptimizationResult.from_dict(result) 
            
            # Store the original total_distance after getting the result
            original_total_distance = result.total_distance
            
            # Ensure result is a proper OptimizationResult object. This handles a hypothetical case where 'result' might still not be a DTO after the above.
            if not isinstance(result, OptimizationResult):
                logger.info("Converting result to OptimizationResult using service method _convert_to_optimization_result")
                result = self._convert_to_optimization_result(result)

            # After conversion, ensure the total_distance is preserved if it was valid
            if original_total_distance is not None and hasattr(result, 'total_distance'):
                if result.total_distance != original_total_distance: # Log if it changed and is being reset
                    logger.info(f"Preserving original total_distance: {original_total_distance} (was {result.total_distance})")
                result.total_distance = original_total_distance

            # Add detailed paths
            if result.status == 'success':
                if use_api_flag:
                    # Use Google Maps for detailed paths
                    logger.info("Adding detailed paths using Google Maps API")
                    try:
                        # Create traffic service first with API key
                        traffic_service = TrafficService(api_key=api_key_to_use)
                        # Then create road graph - separate to ensure the call is made
                        graph = traffic_service.create_road_graph(locations)
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
            
            if result.status == 'success': # Only cache successful results, or based on your policy
                cacheable_result_dict = dataclasses.asdict(result)
                # Define cache timeout (e.g., 1 hour, or from settings)
                cache_timeout_seconds = getattr(settings, 'OPTIMIZATION_RESULT_CACHE_TIMEOUT', 3600) 
                cache.set(cache_key, cacheable_result_dict, timeout=cache_timeout_seconds)
                logger.info(f"Cached OptimizationResult for key: {cache_key} for {cache_timeout_seconds}s")
            # --- End Store in Cache ---

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
                statistics={'error': f"Optimization failed: {str(e)}"}
            )
