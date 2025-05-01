from route_optimizer.core.distance_matrix import DistanceMatrixBuilder

class TrafficService:
    @staticmethod
    def apply_traffic_factors(distance_matrix, traffic_data):
        return DistanceMatrixBuilder.add_traffic_factors(distance_matrix, traffic_data)
