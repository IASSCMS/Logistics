from django.test import TestCase
from collections import namedtuple
from route_optimizer.services.depot_service import DepotService

Location = namedtuple('Location', ['is_depot'])

class DepotServiceTest(TestCase):
    def test_find_depot_index_with_depot(self):
        locations = [Location(False), Location(True), Location(False)]
        self.assertEqual(DepotService.find_depot_index(locations), 1)

    def test_find_depot_index_without_depot(self):
        locations = [Location(False), Location(False)]
        self.assertEqual(DepotService.find_depot_index(locations), 0)
