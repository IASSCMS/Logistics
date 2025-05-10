class RouteStatsService:
    """
    Service for calculating statistics about optimized routes.
    """
    
    @staticmethod
    def add_statistics(result, vehicles):
        """
        Add statistics to the optimization result.
        
        Args:
            result: The optimization result to enrich
            vehicles: List of vehicles used in optimization
        """
        # Initialize statistics
        result['vehicle_costs'] = {}
        result['total_cost'] = 0
        
        # Ensure detailed_routes exists
        if 'detailed_routes' not in result:
            result['detailed_routes'] = []
            if 'routes' in result:
                for route_idx, route in enumerate(result['routes']):
                    vehicle_id = None
                    if 'assigned_vehicles' in result:
                        for v_id, v_route_idx in result['assigned_vehicles'].items():
                            if v_route_idx == route_idx:
                                vehicle_id = v_id
                                break
                    
                    result['detailed_routes'].append({
                        'stops': route,
                        'segments': [],
                        'vehicle_id': vehicle_id
                    })
        
        # Calculate costs for each route
        for route_idx, route in enumerate(result['detailed_routes']):
            # Find vehicle details
            vehicle_id = route.get('vehicle_id')
            vehicle = next((v for v in vehicles if str(v.id) == str(vehicle_id)), None)
            
            # Default distance if not available in segments
            route_distance = 0
            if 'segments' in route:
                route_distance = sum(segment.get('distance', 0) for segment in route['segments'])
            
            # Calculate costs if vehicle is found
            if vehicle:
                fixed_cost = getattr(vehicle, 'fixed_cost', 0)
                cost_per_km = getattr(vehicle, 'cost_per_km', 0)
                variable_cost = route_distance * cost_per_km
                total_vehicle_cost = fixed_cost + variable_cost
                
                result['vehicle_costs'][vehicle_id] = {
                    'fixed_cost': fixed_cost,
                    'variable_cost': variable_cost,
                    'cost': total_vehicle_cost,  # Add 'cost' key for compatibility
                    'total_cost': total_vehicle_cost,
                    'distance': route_distance
                }
                
                result['total_cost'] += total_vehicle_cost
        
        # Calculate total statistics
        total_stops = sum(len(route.get('stops', [])) for route in result['detailed_routes'])
        total_distance = sum(
            sum(segment.get('distance', 0) for segment in route.get('segments', []))
            for route in result['detailed_routes']
        )
        
        # Add summary statistics
        result['summary'] = {
            'total_stops': total_stops,
            'total_distance': total_distance,
            'total_vehicles': len([r for r in result['detailed_routes'] if r.get('vehicle_id')]),
            'total_cost': result['total_cost']
        }
        
        return result
