import logging
logger = logging.getLogger(__name__)

from route_optimizer.core.types_1 import DetailedRoute, OptimizationResult, RouteSegment

class PathAnnotator:
    def __init__(self, path_finder):
        self.path_finder = path_finder
        
    def _add_summary_statistics(self, result, vehicles):
        """
        Add summary statistics to the optimization result.
        
        Args:
            result: The optimization result to enrich
            vehicles: List of vehicles used in optimization
            
        Returns:
            The enriched result with summary statistics
        """
        # Handle both dict and OptimizationResult
        is_dto = isinstance(result, OptimizationResult)
        
        # Ensure detailed_routes exists
        if is_dto:
            if not result.detailed_routes:
                result.detailed_routes = []
                
                # Convert routes to detailed_routes if needed
                if result.routes:
                    for route_idx, route in enumerate(result.routes):
                        # Find vehicle for this route
                        vehicle_id = None
                        if result.assigned_vehicles:
                            for v_id, v_route_idx in result.assigned_vehicles.items():
                                if v_route_idx == route_idx:
                                    vehicle_id = v_id
                                    break
                        
                        detailed_route = DetailedRoute(
                            vehicle_id=vehicle_id or f"unknown_{route_idx}",
                            stops=route,
                            segments=[]
                        )
                        result.detailed_routes.append(vars(detailed_route))
        else:
            # Dict case
            if 'detailed_routes' not in result:
                result['detailed_routes'] = []
                # Convert routes to detailed_routes if needed
                if 'routes' in result:
                    for route_idx, route in enumerate(result['routes']):
                        # Find vehicle for this route
                        vehicle_id = None
                        if 'assigned_vehicles' in result:
                            for v_id, v_route_idx in result['assigned_vehicles'].items():
                                if v_route_idx == route_idx:
                                    vehicle_id = v_id
                                    break
                        
                        result['detailed_routes'].append({
                            'stops': route,
                            'segments': [],
                            'vehicle_id': vehicle_id or f"unknown_{route_idx}"
                        })
        
        # Ensure each route has a 'stops' key
        if is_dto:
            for route in result.detailed_routes:
                if 'stops' not in route:
                    # If no stops but we have segments, create stops from segments
                    if 'segments' in route and route['segments']:
                        segments = route['segments']
                        stops = [segments[0]['from']]
                        for segment in segments:
                            stops.append(segment['to'])
                        route['stops'] = stops
                    else:
                        # Default empty stops
                        route['stops'] = []
        else:
            for route in result['detailed_routes']:
                if 'stops' not in route:
                    # If no stops but we have segments, create stops from segments
                    if 'segments' in route and route['segments']:
                        segments = route['segments']
                        stops = [segments[0]['from']]
                        for segment in segments:
                            stops.append(segment['to'])
                        route['stops'] = stops
                    else:
                        # Default empty stops
                        route['stops'] = []
        
        from route_optimizer.services.route_stats_service import RouteStatsService
        RouteStatsService.add_statistics(result, vehicles)
        # # Replace with our mock
        # PathAnnotator._add_summary_statistics.__globals__['RouteStatsService'] = mock_stats_service
        return result

    def annotate(self, result, graph_or_matrix):
        """
        Annotate routes with detailed path information.
        
        Args:
            result: The optimization result (dict or OptimizationResult)
            graph_or_matrix: Either a graph dictionary or a tuple of (distance_matrix, location_ids)
                
        Returns:
            The annotated result
        """
        # Check if we're given a matrix instead of a graph
        if isinstance(graph_or_matrix, dict) and 'matrix' in graph_or_matrix and 'location_ids' in graph_or_matrix:
            # Convert the matrix to a graph
            matrix = graph_or_matrix['matrix']
            location_ids = graph_or_matrix['location_ids']
            
            # Handle if the matrix is a numpy array
            from route_optimizer.core.distance_matrix import DistanceMatrixBuilder
            graph = DistanceMatrixBuilder.distance_matrix_to_graph(matrix, location_ids)
        else:
            # Already a graph
            graph = graph_or_matrix
        
        is_dto = isinstance(result, OptimizationResult)
        
        # Get routes from either DTO or dict
        if is_dto:
            routes = result.detailed_routes if result.detailed_routes else []
            # If no detailed routes but we have routes, use those
            if not routes and result.routes:
                for route_idx, route in enumerate(result.routes):
                    # Find which vehicle is assigned to this route
                    vehicle_id = None
                    for v_id, v_route_idx in result.assigned_vehicles.items():
                        if v_route_idx == route_idx:
                            vehicle_id = v_id
                            break
                    
                    detailed_route = DetailedRoute(
                        vehicle_id=vehicle_id or f"unknown_{route_idx}",
                        stops=route
                    )
                    routes.append(vars(detailed_route))
                result.detailed_routes = routes
        else:
            # Working with dict
            if 'detailed_routes' not in result:
                result['detailed_routes'] = []
            
            routes = result['detailed_routes']
            # If no detailed routes but we have routes, use those
            if not routes and 'routes' in result:
                for route_idx, route in enumerate(result['routes']):
                    # Find which vehicle is assigned to this route
                    vehicle_id = None
                    if 'assigned_vehicles' in result:
                        for v_id, v_route_idx in result['assigned_vehicles'].items():
                            if v_route_idx == route_idx:
                                vehicle_id = v_id
                                break
                    
                    routes.append({
                        'stops': route,
                        'segments': [],
                        'vehicle_id': vehicle_id or f"unknown_{route_idx}"
                    })
                result['detailed_routes'] = routes
        
        # Process each route
        for route in routes:
            stops = route.get('stops', [])
            segments = []
            
            for i in range(len(stops) - 1):
                from_location = stops[i]
                to_location = stops[i + 1]
                
                try:
                    path, distance = self.path_finder.calculate_shortest_path(graph, from_location, to_location)
                    
                    if path:
                        segment = RouteSegment(
                            from_location=from_location,
                            to_location=to_location,
                            path=path,
                            distance=distance
                        )
                        segments.append(vars(segment))
                except Exception as e:
                    logger.error(f"Error calculating path from {from_location} to {to_location}: {e}")
                    # Add a placeholder segment with error information
                    path, distance = [from_location, to_location], 0.0
                    segments.append({
                        'from_location': from_location,
                        'to_location': to_location,
                        'path': [from_location, to_location],
                        'distance': 0.0,
                        'error': str(e)
                    })
            
            route['segments'] = segments
        
        # Return the updated result
        return result
