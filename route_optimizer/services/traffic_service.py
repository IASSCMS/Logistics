from route_optimizer.core.distance_matrix import DistanceMatrixBuilder

class TrafficService:
    @staticmethod
    def apply_traffic_factors(distance_matrix, traffic_data):
        return DistanceMatrixBuilder.add_traffic_factors(distance_matrix, traffic_data)
    
    def create_road_graph(self, locations):
        """
        Create a road graph from a list of locations.
        
        Args:
            locations: List of Location objects
            
        Returns:
            Graph representation of roads between locations
        """
        # Implementation depends on your graph structure
        # Basic implementation could return a dictionary with nodes and edges
        graph = {'nodes': {}, 'edges': {}}
        for location in locations:
            graph['nodes'][location.id] = location
        
        # Create edges between all nodes
        for loc1 in locations:
            graph['edges'][loc1.id] = {}
            for loc2 in locations:
                if loc1.id != loc2.id:
                    # Calculate distance (you might use Haversine or call a service)
                    graph['edges'][loc1.id][loc2.id] = self._calculate_distance(loc1, loc2)
        
        return graph

