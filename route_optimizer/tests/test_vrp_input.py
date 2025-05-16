import unittest

from route_optimizer.models.vrp_input import VRPInputBuilder, Vehicle, Location, DeliveryTask, VRPCompiler


class TestVRPModel(unittest.TestCase):

    def setUp(self):
        self.builder = VRPInputBuilder()
        self.vehicle = Vehicle(id="V1", depot=Location(6.9, 79.8), capacity=15)
        self.task1 = DeliveryTask(id="D1", pickup=Location(7.0, 80.0), delivery=Location(7.1, 80.1), demand=5)
        self.task2 = DeliveryTask(id="D2", pickup=Location(7.2, 80.2), delivery=Location(7.3, 80.3), demand=10)

    def test_add_vehicle(self):
        self.builder.add_vehicle(self.vehicle)
        self.assertEqual(len(self.builder.vehicles), 1)
        self.assertIn("V1_depot", self.builder.location_labels)

    def test_add_delivery_task(self):
        self.builder.add_delivery_task(self.task1)
        self.assertEqual(len(self.builder.tasks), 1)
        self.assertIn("D1_pickup", self.builder.location_labels)
        self.assertIn("D1_delivery", self.builder.location_labels)

    def test_distance_matrix_growth(self):
        self.builder.add_vehicle(self.vehicle)
        self.builder.add_delivery_task(self.task1)
        self.assertEqual(len(self.builder.distance_matrix), 3)  # depot + pickup + delivery
        for row in self.builder.distance_matrix:
            self.assertEqual(len(row), 3)

    def test_set_distance(self):
        self.builder.add_vehicle(self.vehicle)
        self.builder.add_delivery_task(self.task1)
        self.builder.set_distance(0, 1, 42)
        self.assertEqual(self.builder.distance_matrix[0][1], 42)

    def test_compile_vrp_input(self):
        self.builder.add_vehicle(self.vehicle)
        self.builder.add_delivery_task(self.task1)
        self.builder.add_delivery_task(self.task2)
        vrp_input = VRPCompiler.compile(self.builder)
        vrp_input.validate()

        self.assertEqual(vrp_input.num_vehicles, 1)
        self.assertEqual(len(vrp_input.pickups_deliveries), 2)
        self.assertEqual(len(vrp_input.demands), len(vrp_input.location_ids))
        self.assertIn(5, vrp_input.demands)
        self.assertIn(-5, vrp_input.demands)
        self.assertIn(10, vrp_input.demands)
        self.assertIn(-10, vrp_input.demands)

    def test_task_index_map_consistency(self):
        self.builder.add_vehicle(self.vehicle)
        self.builder.add_delivery_task(self.task1)
        vrp_input = VRPCompiler.compile(self.builder)
        task_id = vrp_input.task_index_map[vrp_input.pickups_deliveries[0][0]][0]
        self.assertEqual(task_id, "D1")


if __name__ == "__main__":
    unittest.main()
