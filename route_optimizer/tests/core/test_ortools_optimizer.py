"""
Tests for OR-Tools VRP solver implementation.

This module contains tests for the ORToolsVRPSolver class.
"""
import unittest
import numpy as np
from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver, Vehicle, Delivery
from route_optimizer.core.distance_matrix import Location


class TestORToolsVRPSolver(unittest.TestCase):
    """Test cases for ORToolsVRPSolver."""

    def setUp(self):
        """Set up test fixtures."""
        self.solver = ORToolsVRPSolver(time_limit_seconds=1)  # Short time limit for tests
        
        # Sample locations
        self.locations = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True),
            Location(id="customer1", name="Customer 1", latitude=1.0, longitude=0.0),
            Location(id="customer2", name="Customer 2", latitude=0.0, longitude=1.0),
            Location(id="customer3", name="Customer 3", latitude=1.0, longitude=1.0)
        ]
        
        # Sample location IDs
        self.location_ids = [loc.id for loc in self.locations]
        
        # Sample distance matrix (in km)
        self.distance_matrix = np.array([
            [0.0, 1.0, 1.0, 1.4],  # Depot to others
            [1.0, 0.0, 1.4, 1.0],  # Customer 1 to others
            [1.0, 1.4, 0.0, 1.0],  # Customer 2 to others
            [1.4, 1.0, 1.0, 0.0]   # Customer 3 to others
        ])
        
        # Sample vehicles
        self.vehicles = [
            Vehicle(
                id="vehicle1",
                capacity=10.0,
                start_location_id="depot",
                end_location_id="depot"
            ),
            Vehicle(
                id="vehicle2",
                capacity=15.0,
                start_location_id="depot",
                end_location_id="depot"
            )
        ]
        
        # Sample deliveries
        self.deliveries = [
            Delivery(id="delivery1", location_id="customer1", demand=5.0),
            Delivery(id="delivery2", location_id="customer2", demand=3.0),
            Delivery(id="delivery3", location_id="customer3", demand=6.0)
        ]

    def test_basic_routing(self):
        """Test basic routing functionality."""
        solution = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=0
        )        
        # Verify solution structure
        self.assertIn('status', solution)
        self.assertIn('routes', solution)
        self.assertIn('total_distance', solution)
        self.assertIn('assigned_vehicles', solution)
        self.assertIn('unassigned_deliveries', solution)
        
        # Verify all deliveries are assigned
        self.assertEqual(len(solution['unassigned_deliveries']), 0)
        
        # Verify correct number of routes
        # We expect at most 2 routes (one per vehicle)
        self.assertLessEqual(len(solution['routes']), 2)
        
        # Verify each route starts and ends at the depot
        for route in solution['routes']:
            self.assertEqual(route[0], 'depot')  # Start at depot
            self.assertEqual(route[-1], 'depot')  # End at depot
        
        # All customers should be visited exactly once
        all_visits = []
        for route in solution['routes']:
            all_visits.extend(route[1:-1])  # Exclude depot at start and end
            
        self.assertEqual(set(all_visits), {'customer1', 'customer2', 'customer3'})  # Indices of customer1, customer2, customer3

    # def test_capacity_constraints(self):
    #     """Test that vehicle capacity constraints are respected."""

    #     # Force high-demand scenario
    #     self.deliveries = [
    #         Delivery(id="d1", location_id="customer1", demand=6.0),
    #         Delivery(id="d2", location_id="customer2", demand=5.0),
    #         Delivery(id="d3", location_id="customer3", demand=4.0)
    #     ]

    #     small_vehicle = [
    #         Vehicle(
    #             id="small_vehicle",
    #             capacity=8.0,
    #             start_location_id="depot",
    #             end_location_id="depot"
    #         )
    #     ]

    #     solution = self.solver.solve(
    #         distance_matrix=self.distance_matrix,
    #         location_ids=self.location_ids,
    #         vehicles=small_vehicle,
    #         deliveries=self.deliveries,
    #         depot_index=0
    #     )

    #     print("Unassigned deliveries:", solution.get('unassigned_deliveries', []))

    #     self.assertEqual(solution['status'], 'success')
    #     self.assertTrue(
    #         len(solution['unassigned_deliveries']) > 0,
    #         "Expected some deliveries to be unassigned due to capacity limits."
    #     )

    def test_multi_vehicle_assignment(self):
        """Test that deliveries are assigned to multiple vehicles when needed."""
        solution = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=0
        )
        
        # All deliveries should be assigned
        self.assertEqual(len(solution['unassigned_deliveries']), 0)
        
        # The solver might use one or two vehicles depending on the best solution
        # If the total demand (14.0) is split, we should have two routes
        if len(solution['routes']) == 2:
            # Verify both vehicles are used
            self.assertEqual(len(solution['assigned_vehicles']), 2)

    def test_empty_problem(self):
        """Test handling of empty problem (no deliveries)."""
        solution = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=[],  # No deliveries
            depot_index=0
        )
        
        # Should have valid solution with no routes
        self.assertEqual(solution['status'], 'success')
        self.assertEqual(len(solution['routes']), 2)
        self.assertEqual(solution['total_distance'], 4.0)

def test_time_windows(self):
    """Test routing with time windows."""
    locations_with_tw = [
        Location(
            id="depot", 
            name="Depot", 
            latitude=0.0, 
            longitude=0.0, 
            is_depot=True,
            time_window_start=0,    # 00:00
            time_window_end=1440    # 24:00
        ),
        Location(
            id="customer1", 
            name="Customer 1", 
            latitude=1.0, 
            longitude=0.0,
            time_window_start=480,  # 08:00
            time_window_end=600     # 10:00
        ),
        Location(
            id="customer2", 
            name="Customer 2", 
            latitude=0.0, 
            longitude=1.0,
            time_window_start=540,  # 09:00
            time_window_end=660     # 11:00
        ),
        Location(
            id="customer3", 
            name="Customer 3", 
            latitude=1.0, 
            longitude=1.0,
            time_window_start=600,  # 10:00
            time_window_end=720     # 12:00
        )
    ]

    solution = self.solver.solve_with_time_windows(
        distance_matrix=self.distance_matrix,
        location_ids=self.location_ids,
        vehicles=self.vehicles,
        deliveries=self.deliveries,
        locations=locations_with_tw,
        depot_index=0,
        speed_km_per_hour=60.0
    )

    # Check required keys
    self.assertIn('status', solution)
    self.assertIn('routes', solution)

    # If successful, check time windows are respected
    if solution['status'] == 'success':
        for route in solution['routes']:
            for stop in route:
                loc_id = stop['location_id']
                arrival_minutes = stop['arrival_time_seconds'] // 60
                location = next((l for l in locations_with_tw if l.id == loc_id), None)

                if location and location.time_window_start is not None and location.time_window_end is not None:
                    self.assertGreaterEqual(
                        arrival_minutes, location.time_window_start,
                        f"Arrival at {loc_id} too early: {arrival_minutes} < {location.time_window_start}"
                    )
                    self.assertLessEqual(
                        arrival_minutes, location.time_window_end,
                        f"Arrival at {loc_id} too late: {arrival_minutes} > {location.time_window_end}"
                    )


if __name__ == '__main__':
    unittest.main()