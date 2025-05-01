from django.test import TestCase
import numpy as np
from route_optimizer.services.traffic_service import TrafficService

class TrafficServiceTest(TestCase):
    def test_apply_traffic_factors(self):
        matrix = np.array([[0, 10], [10, 0]])
        traffic_data = {(0, 1): 1.5, (1, 0): 2.0}
        
        adjusted_matrix = TrafficService.apply_traffic_factors(matrix.copy(), traffic_data)
        
        self.assertEqual(adjusted_matrix[0,1], 15)
        self.assertEqual(adjusted_matrix[1,0], 20)
