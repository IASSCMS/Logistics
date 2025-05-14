from django.test import TestCase
from route_optimizer.services.depot_service import DepotService
from route_optimizer.core.types_1 import Location #

class DepotServiceTest(TestCase):
    def setUp(self):
        self.depot_service = DepotService()
        # Define dummy coordinates, as they are required by the Location DTO
        self.dummy_lat = 0.0
        self.dummy_lon = 0.0
    
    def test_find_depot_index_with_depot(self):
        locations = [
            Location(id='loc1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False), 
            Location(id='depot', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True), 
            Location(id='loc2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False)
        ]
        self.assertEqual(DepotService.find_depot_index(locations), 1)

    def test_find_depot_index_without_depot(self):
        locations = [
            Location(id='loc1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False), 
            Location(id='loc2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False)
        ]
        self.assertEqual(DepotService.find_depot_index(locations), 0)
        
    def test_find_depot_index_empty_list(self):
        locations = []
        self.assertEqual(DepotService.find_depot_index(locations), 0)
    
    def test_get_nearest_depot_with_one_depot(self):
        locations = [
            Location(id='loc1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False), 
            Location(id='depot', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True), 
            Location(id='loc2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False)
        ]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertIsNotNone(depot)
        self.assertEqual(depot.id, 'depot')
        
    def test_get_nearest_depot_with_multiple_depots(self):
        locations = [
            Location(id='loc1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False), 
            Location(id='depot1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True), 
            Location(id='loc2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False),
            Location(id='depot2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True)
        ]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertIsNotNone(depot)
        self.assertEqual(depot.id, 'depot1')  # Should return first depot
        
    def test_get_nearest_depot_without_depot(self):
        locations = [
            Location(id='loc1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False), 
            Location(id='loc2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False)
        ]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertIsNotNone(depot)
        self.assertEqual(depot.id, 'loc1')  # Should return first location
        
    def test_get_nearest_depot_empty_list(self):
        locations = []
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertIsNone(depot)  # Should return None for empty list

    def test_get_nearest_depot_with_only_depots(self):
        locations = [
            Location(id='depot1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True),
            Location(id='depot2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True)
        ]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertIsNotNone(depot)
        self.assertEqual(depot.id, 'depot1') # Should return the first depot

    def test_find_depot_index_with_only_depots(self):
        locations = [
            Location(id='depot1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True),
            Location(id='depot2', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True)
        ]
        self.assertEqual(DepotService.find_depot_index(locations), 0) # Returns index of the first depot

    def test_get_nearest_depot_with_depot_first(self):
        locations = [
            Location(id='depot1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True),
            Location(id='loc1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False)
        ]
        depot = self.depot_service.get_nearest_depot(locations)
        self.assertIsNotNone(depot)
        self.assertEqual(depot.id, 'depot1')

    def test_find_depot_index_with_depot_first(self):
        locations = [
            Location(id='depot1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=True),
            Location(id='loc1', latitude=self.dummy_lat, longitude=self.dummy_lon, is_depot=False)
        ]
        self.assertEqual(DepotService.find_depot_index(locations), 0)

