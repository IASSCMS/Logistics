from django.test import TestCase
from route_optimizer.services.route_stats_service import RouteStatsService
from route_optimizer.core.types_1 import OptimizationResult 
from route_optimizer.models import Vehicle 

class RouteStatsServiceTest(TestCase):
    def setUp(self):
        # For Vehicle dataclass, we need to provide all mandatory fields
        # 'id', 'capacity', 'start_location_id' are mandatory.
        # 'cost_per_km' and 'fixed_cost' are used by the service.
        self.vehicle1_dict = Vehicle(id='1', capacity=100, start_location_id='depot', fixed_cost=100, cost_per_km=10)
        self.vehicle2_dict = Vehicle(id='2', capacity=100, start_location_id='depot', fixed_cost=50, cost_per_km=5)
        self.vehicle3_dict = Vehicle(id='3', capacity=100, start_location_id='depot', fixed_cost=75, cost_per_km=8)
        self.vehicle4_dict = Vehicle(id='4', capacity=100, start_location_id='depot', fixed_cost=60, cost_per_km=6)
        self.vehicle5_dict = Vehicle(id='5', capacity=100, start_location_id='depot', fixed_cost=100, cost_per_km=10)

    # --- Tests for Dictionary Input ---

    def test_add_statistics_dict_with_detailed_routes(self):
        # Test with pre-existing detailed routes (dictionary input)
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
        vehicles = [self.vehicle1_dict]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        self.assertIn('vehicle_costs', result)
        self.assertIn('1', result['vehicle_costs'])
        self.assertEqual(result['vehicle_costs']['1']['fixed_cost'], 100)
        self.assertEqual(result['vehicle_costs']['1']['variable_cost'], 12 * 10)
        self.assertEqual(result['vehicle_costs']['1']['cost'], 100 + (12 * 10))
        self.assertEqual(result['vehicle_costs']['1']['total_cost'], 100 + (12 * 10))
        self.assertEqual(result['vehicle_costs']['1']['distance'], 12)
        
        self.assertEqual(result['total_cost'], 100 + (12 * 10))
        
        self.assertIn('summary', result) # For dict, summary is top-level
        self.assertEqual(result['summary']['total_stops'], 3)
        self.assertEqual(result['summary']['total_distance'], 12)
        self.assertEqual(result['summary']['total_vehicles'], 1)
        self.assertEqual(result['summary']['total_cost'], 100 + (12 * 10))
        
        self.assertIn('statistics', result) # Also check statistics dict
        self.assertIn('summary', result['statistics'])
        self.assertEqual(result['statistics']['summary']['total_cost'], 100 + (12 * 10))


    def test_add_statistics_dict_from_routes(self):
        # Test creation of detailed_routes from routes (dictionary input)
        result = {
            'assigned_vehicles': {'2': 0},
            'routes': [['D', 'E', 'F']] # simple routes
        }
        vehicles = [self.vehicle2_dict]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        self.assertIn('detailed_routes', result)
        self.assertEqual(len(result['detailed_routes']), 1)
        self.assertEqual(result['detailed_routes'][0]['stops'], ['D', 'E', 'F'])
        self.assertEqual(result['detailed_routes'][0]['vehicle_id'], '2')
        self.assertEqual(len(result['detailed_routes'][0]['segments']), 0) # No segments created from simple routes
        
        self.assertIn('vehicle_costs', result)
        self.assertIn('2', result['vehicle_costs'])
        self.assertEqual(result['vehicle_costs']['2']['fixed_cost'], 50)
        self.assertEqual(result['vehicle_costs']['2']['variable_cost'], 0) # No distance from segments
        self.assertEqual(result['vehicle_costs']['2']['cost'], 50)
        
        self.assertEqual(result['summary']['total_stops'], 3)
        self.assertEqual(result['summary']['total_vehicles'], 1)
        self.assertEqual(result['summary']['total_distance'], 0) # No segment distances


    def test_add_statistics_dict_multiple_vehicles(self):
        # Test with multiple vehicles (dictionary input)
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
        vehicles = [self.vehicle3_dict, self.vehicle4_dict]
        
        RouteStatsService.add_statistics(result, vehicles)
        
        self.assertEqual(result['total_cost'], (75 + 10*8) + (60 + 20*6)) # 155 + 180 = 335
        self.assertEqual(result['vehicle_costs']['3']['cost'], 155)
        self.assertEqual(result['vehicle_costs']['4']['cost'], 180)
        
        self.assertEqual(result['summary']['total_stops'], 5) # 2 + 3
        self.assertEqual(result['summary']['total_distance'], 30) # 10 + 20
        self.assertEqual(result['summary']['total_vehicles'], 2)

    def test_add_statistics_dict_missing_vehicle(self):
        # Test handling of routes with no matching vehicle (dictionary input)
        result = {
            'detailed_routes': [
                {
                    'vehicle_id': 'unknown_vehicle', # This vehicle is not in `vehicles` list
                    'stops': ['L', 'M'], 
                    'segments': [{'distance': 15}]
                }
            ]
        }
        vehicles = [self.vehicle5_dict] # Only vehicle '5' is known
        
        RouteStatsService.add_statistics(result, vehicles)
        
        self.assertEqual(result['total_cost'], 0) # No cost if vehicle not found
        self.assertEqual(len(result['vehicle_costs']), 0)
        
        self.assertEqual(result['summary']['total_stops'], 2)
        self.assertEqual(result['summary']['total_distance'], 15) # Distance is summed regardless of vehicle match for costs
        self.assertEqual(result['summary']['total_vehicles'], 1) # Counts routes with vehicle_id


    def test_add_statistics_dict_empty_result(self):
        # Test with empty result (dictionary input)
        result = {}
        vehicles = []
        
        RouteStatsService.add_statistics(result, vehicles)
        
        self.assertIn('vehicle_costs', result)
        self.assertEqual(result['total_cost'], 0)
        self.assertIn('detailed_routes', result)
        self.assertEqual(len(result['detailed_routes']), 0)
        self.assertIn('summary', result)
        self.assertEqual(result['summary']['total_stops'], 0)
        self.assertEqual(result['summary']['total_distance'], 0)
        self.assertEqual(result['summary']['total_vehicles'], 0)
        self.assertEqual(result['summary']['total_cost'], 0)

    # --- Tests for OptimizationResult DTO Input ---

    def test_add_statistics_dto_with_detailed_routes(self):
        # Test with pre-existing detailed routes (OptimizationResult DTO input)
        result_dto = OptimizationResult(
            status='success',
            assigned_vehicles={'1': 0},
            detailed_routes=[
                {
                    'vehicle_id': '1',
                    'stops': ['A', 'B', 'C'], 
                    'segments': [{'distance': 5}, {'distance': 7}]
                }
            ]
        )
        vehicles = [self.vehicle1_dict]
        
        RouteStatsService.add_statistics(result_dto, vehicles)
        
        self.assertIn('vehicle_costs', result_dto.statistics)
        self.assertIn('1', result_dto.statistics['vehicle_costs'])
        self.assertEqual(result_dto.statistics['vehicle_costs']['1']['fixed_cost'], 100)
        self.assertEqual(result_dto.statistics['vehicle_costs']['1']['variable_cost'], 12 * 10)
        self.assertEqual(result_dto.statistics['vehicle_costs']['1']['cost'], 100 + (12 * 10))
        self.assertEqual(result_dto.statistics['vehicle_costs']['1']['distance'], 12)
        
        self.assertEqual(result_dto.total_cost, 100 + (12 * 10)) # DTO's total_cost attribute
        
        self.assertIn('summary', result_dto.statistics)
        self.assertEqual(result_dto.statistics['summary']['total_stops'], 3)
        self.assertEqual(result_dto.statistics['summary']['total_distance'], 12)
        self.assertEqual(result_dto.statistics['summary']['total_vehicles'], 1)
        self.assertEqual(result_dto.statistics['summary']['total_cost'], 100 + (12 * 10))

    def test_add_statistics_dto_from_routes(self):
        # Test creation of detailed_routes from routes (OptimizationResult DTO input)
        result_dto = OptimizationResult(
            status='success',
            assigned_vehicles={'2': 0},
            routes=[['D', 'E', 'F']] # simple routes, detailed_routes will be auto-populated
        )
        vehicles = [self.vehicle2_dict]
        
        RouteStatsService.add_statistics(result_dto, vehicles)
        
        self.assertIsNotNone(result_dto.detailed_routes)
        self.assertEqual(len(result_dto.detailed_routes), 1)
        self.assertEqual(result_dto.detailed_routes[0]['stops'], ['D', 'E', 'F'])
        self.assertEqual(result_dto.detailed_routes[0]['vehicle_id'], '2')
        self.assertEqual(len(result_dto.detailed_routes[0]['segments']), 0)
        
        self.assertIn('vehicle_costs', result_dto.statistics)
        self.assertIn('2', result_dto.statistics['vehicle_costs'])
        self.assertEqual(result_dto.statistics['vehicle_costs']['2']['fixed_cost'], 50)
        self.assertEqual(result_dto.statistics['vehicle_costs']['2']['variable_cost'], 0)
        self.assertEqual(result_dto.statistics['vehicle_costs']['2']['cost'], 50)
        
        self.assertEqual(result_dto.total_cost, 50)
        
        self.assertIn('summary', result_dto.statistics)
        self.assertEqual(result_dto.statistics['summary']['total_stops'], 3)
        self.assertEqual(result_dto.statistics['summary']['total_vehicles'], 1)
        self.assertEqual(result_dto.statistics['summary']['total_distance'], 0)

    def test_add_statistics_dto_multiple_vehicles(self):
        # Test with multiple vehicles (OptimizationResult DTO input)
        result_dto = OptimizationResult(
            status='success',
            assigned_vehicles={'3': 0, '4': 1},
            detailed_routes=[
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
        )
        vehicles = [self.vehicle3_dict, self.vehicle4_dict]
        
        RouteStatsService.add_statistics(result_dto, vehicles)
        
        self.assertEqual(result_dto.total_cost, 335)
        self.assertEqual(result_dto.statistics['vehicle_costs']['3']['cost'], 155)
        self.assertEqual(result_dto.statistics['vehicle_costs']['4']['cost'], 180)
        
        self.assertEqual(result_dto.statistics['summary']['total_stops'], 5)
        self.assertEqual(result_dto.statistics['summary']['total_distance'], 30)
        self.assertEqual(result_dto.statistics['summary']['total_vehicles'], 2)
        self.assertEqual(result_dto.statistics['summary']['total_cost'], 335)

    def test_add_statistics_dto_missing_vehicle(self):
        # Test handling of routes with no matching vehicle (OptimizationResult DTO input)
        result_dto = OptimizationResult(
            status='success',
            detailed_routes=[
                {
                    'vehicle_id': 'unknown_vehicle',
                    'stops': ['L', 'M'], 
                    'segments': [{'distance': 15}]
                }
            ]
        )
        vehicles = [self.vehicle5_dict]
        
        RouteStatsService.add_statistics(result_dto, vehicles)
        
        self.assertEqual(result_dto.total_cost, 0)
        self.assertEqual(len(result_dto.statistics.get('vehicle_costs', {})), 0)
        
        self.assertIn('summary', result_dto.statistics)
        self.assertEqual(result_dto.statistics['summary']['total_stops'], 2)
        self.assertEqual(result_dto.statistics['summary']['total_distance'], 15)
        self.assertEqual(result_dto.statistics['summary']['total_vehicles'], 1)
        self.assertEqual(result_dto.statistics['summary']['total_cost'], 0)

    def test_add_statistics_dto_empty_result(self):
        # Test with empty/minimal OptimizationResult DTO input
        result_dto = OptimizationResult(status='success') # Minimal DTO
        vehicles = []
        
        RouteStatsService.add_statistics(result_dto, vehicles)
        
        self.assertIsNotNone(result_dto.statistics)
        self.assertIn('vehicle_costs', result_dto.statistics)
        self.assertEqual(len(result_dto.statistics['vehicle_costs']), 0)
        self.assertEqual(result_dto.total_cost, 0.0)
        self.assertIsNotNone(result_dto.detailed_routes)
        self.assertEqual(len(result_dto.detailed_routes), 0)
        
        self.assertIn('summary', result_dto.statistics)
        self.assertEqual(result_dto.statistics['summary']['total_stops'], 0)
        self.assertEqual(result_dto.statistics['summary']['total_distance'], 0)
        self.assertEqual(result_dto.statistics['summary']['total_vehicles'], 0)
        self.assertEqual(result_dto.statistics['summary']['total_cost'], 0)

    def test_add_statistics_dto_no_segments(self):
        # Test DTO input where detailed_routes exist but have no segments
        result_dto = OptimizationResult(
            status='success',
            assigned_vehicles={'1': 0},
            detailed_routes=[
                {
                    'vehicle_id': '1',
                    'stops': ['A', 'B', 'C'], 
                    'segments': [] # No segments, so distance should be 0
                }
            ]
        )
        vehicles = [self.vehicle1_dict]
        
        RouteStatsService.add_statistics(result_dto, vehicles)
        
        self.assertEqual(result_dto.statistics['vehicle_costs']['1']['distance'], 0)
        self.assertEqual(result_dto.statistics['vehicle_costs']['1']['variable_cost'], 0)
        self.assertEqual(result_dto.statistics['vehicle_costs']['1']['cost'], self.vehicle1_dict.fixed_cost)
        self.assertEqual(result_dto.total_cost, self.vehicle1_dict.fixed_cost)
        self.assertEqual(result_dto.statistics['summary']['total_distance'], 0)
        self.assertEqual(result_dto.statistics['summary']['total_cost'], self.vehicle1_dict.fixed_cost)