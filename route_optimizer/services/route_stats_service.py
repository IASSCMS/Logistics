class RouteStatsService:
    @staticmethod
    def add_statistics(result, vehicles):
        vehicle_costs = {}
        total_cost = 0

        for vehicle_id, route_idx in result['assigned_vehicles'].items():
            vehicle = next((v for v in vehicles if v.id == vehicle_id), None)
            if vehicle:
                route_distance = sum(
                    segment['distance']
                    for segment in result['detailed_routes'][route_idx]['segments']
                )
                cost = vehicle.fixed_cost + (route_distance * vehicle.cost_per_km)
                vehicle_costs[vehicle_id] = {
                    'distance': route_distance,
                    'cost': cost
                }
                total_cost += cost

        result['vehicle_costs'] = vehicle_costs
        result['total_cost'] = total_cost

        total_stops = sum(len(route['stops']) for route in result['detailed_routes'])
        result['total_stops'] = total_stops
        
        if total_stops > 0:
            result['avg_distance_per_stop'] = result['total_distance'] / total_stops

        result['vehicles_used'] = len(result['assigned_vehicles'])
        result['vehicles_unused'] = len(vehicles) - result['vehicles_used']
