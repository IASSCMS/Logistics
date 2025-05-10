from django.test import TestCase
from collections import namedtuple
from route_optimizer.services.route_stats_service import RouteStatsService

Vehicle = namedtuple('Vehicle', ['id', 'fixed_cost', 'cost_per_km'])

class RouteStatsServiceTest(TestCase):
    def test_add_statistics_with_detailed_routes(self):
        # Test with pre-existing detailed routes
        result = {
            'assigned_vehicles': {'1': 0},
            'detailed_routes': [
                {
                    'vehicle_id': '1',
                    'stops': ['A', 'B', 'C'], 
                    'segments': [{'distance': 5}, {'distance': 7}]
                }
            ]
        }
        vehicles = [Vehicle(id='1', fixed_cost=100, cost_per_km=10)]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        # Check vehicle costs
        self.assertIn('vehicle_costs', result)
        self.assertIn('1', result['vehicle_costs'])
        self.assertEqual(result['vehicle_costs']['1']['fixed_cost'], 100)
        self.assertEqual(result['vehicle_costs']['1']['variable_cost'], 12 * 10)
        self.assertEqual(result['vehicle_costs']['1']['total_cost'], 100 + (12 * 10))
        self.assertEqual(result['vehicle_costs']['1']['distance'], 12)
        
        # Check total cost
        self.assertEqual(result['total_cost'], 100 + (12 * 10))
        
        # Check summary statistics
        self.assertIn('summary', result)
        self.assertEqual(result['summary']['total_stops'], 3)
        self.assertEqual(result['summary']['total_distance'], 12)
        self.assertEqual(result['summary']['total_vehicles'], 1)
        self.assertEqual(result['summary']['total_cost'], 100 + (12 * 10))

    def test_add_statistics_from_routes(self):
        # Test creation of detailed_routes from routes
        result = {
            'assigned_vehicles': {'2': 0},
            'routes': [['D', 'E', 'F']]
        }
        vehicles = [Vehicle(id='2', fixed_cost=50, cost_per_km=5)]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        # Check detailed_routes creation
        self.assertIn('detailed_routes', result)
        self.assertEqual(len(result['detailed_routes']), 1)
        self.assertEqual(result['detailed_routes'][0]['stops'], ['D', 'E', 'F'])
        self.assertEqual(result['detailed_routes'][0]['vehicle_id'], '2')
        
        # Check vehicle costs (with zero distance since no segments)
        self.assertIn('vehicle_costs', result)
        self.assertIn('2', result['vehicle_costs'])
        self.assertEqual(result['vehicle_costs']['2']['fixed_cost'], 50)
        self.assertEqual(result['vehicle_costs']['2']['variable_cost'], 0)
        self.assertEqual(result['vehicle_costs']['2']['total_cost'], 50)
        
        # Check summary statistics
        self.assertEqual(result['summary']['total_stops'], 3)
        self.assertEqual(result['summary']['total_vehicles'], 1)

    def test_add_statistics_multiple_vehicles(self):
        # Test with multiple vehicles
        result = {
            'assigned_vehicles': {'3': 0, '4': 1},
            'detailed_routes': [
                {
                    'vehicle_id': '3',
                    'stops': ['G', 'H'], 
                    'segments': [{'distance': 10}]
                },
                {
                    'vehicle_id': '4',
                    'stops': ['I', 'J', 'K'], 
                    'segments': [{'distance': 8}, {'distance': 12}]
                }
            ]
        }
        vehicles = [
            Vehicle(id='3', fixed_cost=75, cost_per_km=8),
            Vehicle(id='4', fixed_cost=60, cost_per_km=6)
        ]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        # Check total cost (75 + 10*8) + (60 + 20*6) = 155 + 180 = 335
        self.assertEqual(result['total_cost'], 335)
        
        # Check vehicle costs
        self.assertEqual(result['vehicle_costs']['3']['total_cost'], 155)
        self.assertEqual(result['vehicle_costs']['4']['total_cost'], 180)
        
        # Check summary statistics
        self.assertEqual(result['summary']['total_stops'], 5)
        self.assertEqual(result['summary']['total_distance'], 30)
        self.assertEqual(result['summary']['total_vehicles'], 2)

    def test_add_statistics_missing_vehicle(self):
        # Test handling of routes with no matching vehicle
        result = {
            'detailed_routes': [
                {
                    'vehicle_id': 'unknown',
                    'stops': ['L', 'M'], 
                    'segments': [{'distance': 15}]
                }
            ]
        }
        vehicles = [Vehicle(id='5', fixed_cost=100, cost_per_km=10)]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        # Check that we don't have costs for the unknown vehicle
        self.assertEqual(result['total_cost'], 0)
        self.assertEqual(len(result['vehicle_costs']), 0)
        
        # Check summary statistics still count the route
        self.assertEqual(result['summary']['total_stops'], 2)
        self.assertEqual(result['summary']['total_distance'], 15)
        self.assertEqual(result['summary']['total_vehicles'], 1)

    def test_add_statistics_empty_result(self):
        # Test with empty result
        result = {}
        vehicles = []
        
        RouteStatsService.add_statistics(result, vehicles)
        
        # Check all expected keys are present with default values
        self.assertIn('vehicle_costs', result)
        self.assertEqual(result['total_cost'], 0)
        self.assertIn('detailed_routes', result)
        self.assertIn('summary', result)
        self.assertEqual(result['summary']['total_stops'], 0)
        self.assertEqual(result['summary']['total_distance'], 0)
        self.assertEqual(result['summary']['total_vehicles'], 0)
        self.assertEqual(result['summary']['total_cost'], 0)