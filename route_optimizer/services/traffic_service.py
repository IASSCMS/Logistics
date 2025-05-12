from route_optimizer.core.distance_matrix import DistanceMatrixBuilder

class TrafficService:
    def __init__(self, api_key=None):
        """
        Initialize the traffic service with optional API key.
        
        Args:
            api_key: API key for external services if applicable
        """
        self.api_key = api_key
    
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
    
    def _calculate_distance(self, loc1, loc2):
        """
        Calculate the distance between two locations.
        Could use the API key for external services if needed.
        """
        # Basic implementation - could be enhanced to use API if self.api_key is available
        if hasattr(loc1, 'latitude') and hasattr(loc1, 'longitude') and \
           hasattr(loc2, 'latitude') and hasattr(loc2, 'longitude'):
            # Simple Euclidean distance as a fallback
            dx = loc1.longitude - loc2.longitude
            dy = loc1.latitude - loc2.latitude
            return (dx**2 + dy**2)**0.5
        return 0.0