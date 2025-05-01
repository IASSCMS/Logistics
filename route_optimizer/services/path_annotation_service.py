class PathAnnotator:
    def __init__(self, path_finder):
        self.path_finder = path_finder

    def annotate(self, result, graph):
        detailed_routes = []
        
        for route in result['routes']:
            detailed_route = {
                'stops': route,
                'segments': []
            }
            
            for i in range(len(route) - 1):
                from_location = route[i]
                to_location = route[i + 1]
                
                path, distance = self.path_finder.calculate_shortest_path(graph, from_location, to_location)
                
                if path:
                    detailed_route['segments'].append({
                        'from': from_location,
                        'to': to_location,
                        'path': path,
                        'distance': distance
                    })
            detailed_routes.append(detailed_route)
        
        result['detailed_routes'] = detailed_routes
