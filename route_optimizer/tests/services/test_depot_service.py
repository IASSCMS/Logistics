from django.test import TestCase
from collections import namedtuple
from route_optimizer.services.depot_service import DepotService

# Expanded namedtuple to include id for easier identification in tests
Location = namedtuple('Location', ['id', 'is_depot'])

class DepotServiceTest(TestCase):
    def setUp(self):
        self.depot_service = DepotService()
    
    def test_find_depot_index_with_depot(self):
        locations = [Location('loc1', False), Location('depot', True), Location('loc2', False)]
        self.assertEqual(DepotService.find_depot_index(locations), 1)

    def test_find_depot_index_without_depot(self):
        locations = [Location('loc1', False), Location('loc2', False)]
        self.assertEqual(DepotService.find_depot_index(locations), 0)
        
    def test_find_depot_index_empty_list(self):
        locations = []
        self.assertEqual(DepotService.find_depot_index(locations), 0)
    
    def test_get_nearest_depot_with_one_depot(self):
        locations = [Location('loc1', False), Location('depot', True), Location('loc2', False)]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertEqual(depot.id, 'depot')
        
    def test_get_nearest_depot_with_multiple_depots(self):
        locations = [
            Location('loc1', False), 
            Location('depot1', True), 
            Location('loc2', False),
            Location('depot2', True)
        ]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertEqual(depot.id, 'depot1')  # Should return first depot
        
    def test_get_nearest_depot_without_depot(self):
        locations = [Location('loc1', False), Location('loc2', False)]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertEqual(depot.id, 'loc1')  # Should return first location
        
    def test_get_nearest_depot_empty_list(self):
        locations = []
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertIsNone(depot)  # Should return None for empty list