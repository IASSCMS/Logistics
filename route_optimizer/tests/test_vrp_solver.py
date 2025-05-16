import unittest
from route_optimizer.models.vrp_input import VRPInputBuilder, VRPCompiler, Location, Vehicle, DeliveryTask
from route_optimizer.services.vrp_solver import solve_cvrp


class TestSolveCVRP(unittest.TestCase):
    def test_basic_functionality(self):
        # Define locations
        depot = Location(0, 0)
        pickup = Location(1, 0)
        delivery = Location(2, 0)

        # Create vehicle and task
        vehicle = Vehicle(id="V1", depot=depot, capacity=10)
        task = DeliveryTask(id="T1", pickup=pickup, delivery=delivery, demand=5)

        # Build VRP input
        builder = VRPInputBuilder()
        builder.add_vehicle(vehicle)
        builder.add_delivery_task(task)

        # Set distances
        builder.set_distance(0, 1, 10)
        builder.set_distance(1, 2, 10)
        builder.set_distance(2, 0, 10)

        vrp_input = VRPCompiler.compile(builder)

        # Solve CVRP
        result = solve_cvrp(vrp_input)

        # Assertions
        self.assertEqual(result["status"], "success")
        self.assertIn("routes", result)
        self.assertIn("total_distance", result)
        self.assertEqual(len(result["routes"]), 1)
        self.assertEqual(result["total_distance"], 30)

    def test_multiple_vehicles_and_tasks(self):
        # Define locations
        depot1 = Location(0, 0)
        depot2 = Location(0, 1)
        pickup1 = Location(1, 0)
        delivery1 = Location(2, 0)
        pickup2 = Location(1, 1)
        delivery2 = Location(2, 1)

        # Create vehicles and tasks
        vehicle1 = Vehicle(id="V1", depot=depot1, capacity=10)
        vehicle2 = Vehicle(id="V2", depot=depot2, capacity=10)
        task1 = DeliveryTask(id="T1", pickup=pickup1, delivery=delivery1, demand=5)
        task2 = DeliveryTask(id="T2", pickup=pickup2, delivery=delivery2, demand=5)

        # Build VRP input
        builder = VRPInputBuilder()
        builder.add_vehicle(vehicle1)
        builder.add_vehicle(vehicle2)
        builder.add_delivery_task(task1)
        builder.add_delivery_task(task2)

        # Set distances (symmetric for simplicity)
        num_locations = len(builder.locations)
        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    builder.set_distance(i, j, 10)

        vrp_input = VRPCompiler.compile(builder)

        # Solve CVRP
        result = solve_cvrp(vrp_input)
        print(result)
        # Assertions
        self.assertEqual(result["status"], "success")
        self.assertLessEqual(len(result["routes"]), 2)
        self.assertLessEqual(result["total_distance"], 60)

    def test_insufficient_vehicle_capacity(self):
        # Define locations
        depot = Location(0, 0)
        pickup = Location(1, 0)
        delivery = Location(2, 0)

        # Create vehicle and task
        vehicle = Vehicle(id="V1", depot=depot, capacity=5)
        task = DeliveryTask(id="T1", pickup=pickup, delivery=delivery, demand=10)

        # Build VRP input
        builder = VRPInputBuilder()
        builder.add_vehicle(vehicle)
        builder.add_delivery_task(task)

        # Set distances
        builder.set_distance(0, 1, 10)
        builder.set_distance(1, 2, 10)
        builder.set_distance(2, 0, 10)

        vrp_input = VRPCompiler.compile(builder)

        # Solve CVRP
        result = solve_cvrp(vrp_input)

        # Assertions
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)

    def test_zero_demand_task(self):
        depot = Location(0, 0)
        pickup = Location(1, 1)
        delivery = Location(2, 2)

        vehicle = Vehicle(id="V1", depot=depot, capacity=5)
        task = DeliveryTask(id="T1", pickup=pickup, delivery=delivery, demand=0)

        builder = VRPInputBuilder()
        builder.add_vehicle(vehicle)
        builder.add_delivery_task(task)

        builder.set_distance(0, 1, 5)
        builder.set_distance(1, 2, 5)
        builder.set_distance(2, 0, 5)

        vrp_input = VRPCompiler.compile(builder)
        result = solve_cvrp(vrp_input)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["routes"]), 1)

    def test_no_tasks(self):
        depot = Location(0, 0)
        vehicle = Vehicle(id="V1", depot=depot, capacity=10)

        builder = VRPInputBuilder()
        builder.add_vehicle(vehicle)

        vrp_input = VRPCompiler.compile(builder)
        result = solve_cvrp(vrp_input)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["routes"]), 0)  # No tasks, so no useful routes
