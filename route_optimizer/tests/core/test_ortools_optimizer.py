import unittest
import numpy as np

from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver, Vehicle, Delivery
from route_optimizer.core.distance_matrix import Location


class TestORToolsVRPSolver(unittest.TestCase):

    def setUp(self):
        self.solver = ORToolsVRPSolver(time_limit_seconds=2)

        self.locations = [
            Location(id="depot", name="Depot", latitude=0, longitude=0, is_depot=True),
            Location(id="A", name="Customer A", latitude=1, longitude=0),
            Location(id="B", name="Customer B", latitude=0, longitude=1),
            Location(id="C", name="Customer C", latitude=1, longitude=1),
        ]

        self.location_ids = [loc.id for loc in self.locations]
        self.distance_matrix = np.array([
            [0.0, 1.0, 1.2, 1.5],
            [1.0, 0.0, 1.3, 1.4],
            [1.2, 1.3, 0.0, 1.0],
            [1.5, 1.4, 1.0, 0.0],
        ])

        self.vehicles = [
            Vehicle(id="v1", capacity=8.0, start_location_id="depot", end_location_id="depot"),
            Vehicle(id="v2", capacity=8.0, start_location_id="depot", end_location_id="depot")
        ]

    def test_basic_solution(self):
        deliveries = [
            Delivery(id="d1", location_id="A", demand=4.0),
            Delivery(id="d2", location_id="B", demand=3.0)
        ]
        res = self.solver.solve(self.distance_matrix, self.location_ids, self.vehicles, deliveries, depot_index=0)
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["unassigned_deliveries"]), 0)

    def test_capacity_exceeded(self):
        deliveries = [
            Delivery(id="d1", location_id="A", demand=10.0),
            Delivery(id="d2", location_id="B", demand=9.0)
        ]
        res = self.solver.solve(self.distance_matrix, self.location_ids, self.vehicles, deliveries, depot_index=0)
        self.assertEqual(res["status"], "failed")

    def test_no_deliveries(self):
        res = self.solver.solve(self.distance_matrix, self.location_ids, self.vehicles, [], depot_index=0)
        self.assertEqual(res["status"], "success")
        for r in res["routes"]:
            self.assertEqual(r[0], "depot")
            self.assertEqual(r[-1], "depot")

    def test_pickup_before_delivery_constraint(self):
        # Manually inject pickup-delivery pair
        self.solver.pickup_delivery_pairs = [("A", "B")]
        deliveries = [
            Delivery(id="pickup", location_id="A", demand=2.0, is_pickup=True),
            Delivery(id="drop", location_id="B", demand=2.0)
        ]
        res = self.solver.solve_with_time_windows(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=deliveries,
            locations=self.locations,
            depot_index=0,
            speed_km_per_hour=60.0
        )
        self.assertEqual(res["status"], "success")
        # Assert pickup comes before delivery in route
        for route in res["routes"]:
            steps = [step["location_id"] for step in route]
            if "A" in steps and "B" in steps:
                self.assertLess(steps.index("A"), steps.index("B"))

    def test_high_demand_and_splitting(self):
        deliveries = [
            Delivery(id="d1", location_id="A", demand=4.0),
            Delivery(id="d2", location_id="B", demand=4.0),
            Delivery(id="d3", location_id="C", demand=6.0)
        ]
        res = self.solver.solve(self.distance_matrix, self.location_ids, self.vehicles, deliveries, depot_index=0)
        self.assertEqual(res["status"], "success")
        self.assertTrue(len(res["routes"]) <= 2)

    def test_route_round_trip(self):
        deliveries = [
            Delivery(id="d1", location_id="A", demand=2.0),
            Delivery(id="d2", location_id="B", demand=2.0)
        ]
        res = self.solver.solve(self.distance_matrix, self.location_ids, self.vehicles, deliveries, depot_index=0)
        for route in res["routes"]:
            self.assertEqual(route[0], "depot")
            self.assertEqual(route[-1], "depot")


if __name__ == "__main__":
    unittest.main()
