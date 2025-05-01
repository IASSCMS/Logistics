from django.test import TestCase
from collections import namedtuple
from route_optimizer.services.route_stats_service import RouteStatsService

Vehicle = namedtuple('Vehicle', ['id', 'fixed_cost', 'cost_per_km'])

class RouteStatsServiceTest(TestCase):
    def test_add_statistics(self):
        result = {
            'assigned_vehicles': {1: 0},
            'detailed_routes': [
                {'stops': ['A', 'B'], 'segments': [{'distance': 5}, {'distance': 7}]}
            ],
            'total_distance': 12
        }
        vehicles = [Vehicle(id=1, fixed_cost=100, cost_per_km=10)]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        self.assertIn('vehicle_costs', result)
        self.assertEqual(result['total_cost'], 100 + (12 * 10))
