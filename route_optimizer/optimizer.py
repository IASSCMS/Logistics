
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

class RouteOptimizer:
    def __init__(self, locations, vehicle_capacities):
        self.locations = locations
        self.vehicle_capacities = vehicle_capacities
        self.depot_index = 0
        #print("RouteOptimizer initialized with locations:", [loc['id'] for loc in locations])

    def validate_inputs(self):
        if not self.locations:
            raise ValueError("No delivery locations provided")
        if self.locations[0]['load'] != 0:
            raise ValueError("Depot must have 0 load")
        if len(self.vehicle_capacities) < 1:
            raise ValueError("At least one vehicle required")

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance = R * c * 1000  # Convert to meters
        rounded_distance = round(distance)
        return rounded_distance

    def create_distance_matrix(self):
        matrix = []
        for loc1 in self.locations:
            row = []
            lat1, lon1 = loc1['coordinates']
            for loc2 in self.locations:
                lat2, lon2 = loc2['coordinates']
                distance = self.haversine_distance(lat1, lon1, lat2, lon2)
                row.append(distance)
            matrix.append(row)
        print("Distance Matrix:", matrix)
        return matrix

    def solve(self):
        self.validate_inputs()
        data = {
            'distance_matrix': self.create_distance_matrix(),
            'loads': [loc['load'] for loc in self.locations],
            'vehicle_capacities': self.vehicle_capacities,
            'num_vehicles': len(self.vehicle_capacities),
            'depot': self.depot_index
        }

        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            distance = int(data['distance_matrix'][from_node][to_node])
            return distance

        transit_callback_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

        def load_callback(from_index):
            return data['loads'][manager.IndexToNode(from_index)]

        load_callback_idx = routing.RegisterUnaryTransitCallback(load_callback)
        routing.AddDimensionWithVehicleCapacity(
            load_callback_idx,
            0,  # null capacity slack
            data['vehicle_capacities'],
            True,
            'Capacity'
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC
        )
        search_params.time_limit.FromSeconds(10)
        solution = routing.SolveWithParameters(search_params)
        return self.format_solution(data, manager, routing, solution)

    def format_solution(self, data, manager, routing, solution):
        if not solution:
            return None

        routes = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            route_load = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(self.locations[node]['id'])
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(index)
                distance = data['distance_matrix'][node][next_node]
                route_distance += distance
                route_load += data['loads'][node]

            route.append(self.locations[self.depot_index]['id'])
            routes.append({
                'vehicle_id': vehicle_id,
                'route': route,
                'distance': route_distance,
                'load': route_load
            })
        
        return routes