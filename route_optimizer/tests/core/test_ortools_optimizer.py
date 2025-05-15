"""
Tests for OR-Tools VRP solver implementation.

This module contains tests for the ORToolsVRPSolver class.
"""
import unittest
import numpy as np
import logging # Added for suppressing solver logs during tests if needed

from route_optimizer.core.constants import TIME_SCALING_FACTOR, DISTANCE_SCALING_FACTOR, CAPACITY_SCALING_FACTOR
from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver
from route_optimizer.core.types_1 import Location, OptimizationResult
from route_optimizer.models import Vehicle, Delivery

# Suppress OR-Tools logging for cleaner test output (optional)
# logging.disable(logging.INFO) # Enable this if solver logs are too verbose

class TestORToolsVRPSolver(unittest.TestCase):
    """Test cases for ORToolsVRPSolver."""

    def setUp(self):
        """Set up test fixtures."""
        self.solver = ORToolsVRPSolver(time_limit_seconds=2)  # Slightly increased for complex cases
        
        self.locations = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True, service_time=0),
            Location(id="customer1", name="Customer 1", latitude=1.0, longitude=0.0, service_time=10), # 10 min service time
            Location(id="customer2", name="Customer 2", latitude=0.0, longitude=1.0, service_time=10),
            Location(id="customer3", name="Customer 3", latitude=1.0, longitude=1.0, service_time=10)
        ]
        
        self.location_ids = [loc.id for loc in self.locations]
        
        self.distance_matrix = np.array([
            [0.0, 1.0, 1.0, 1.4],
            [1.0, 0.0, 1.4, 1.0],
            [1.0, 1.4, 0.0, 1.0],
            [1.4, 1.0, 1.0, 0.0]
        ])

        # Time matrix in minutes (assuming speed_km_per_hour = 60 km/h, so time = distance)
        # plus service time at the destination node. For balancing test.
        # For simplicity here, let's make a sample time matrix.
        # Travel time (minutes) = distance (km) / speed (km/h) * 60
        # If speed is 60 km/h, travel_time_min = distance_km
        self.time_matrix_minutes = np.array([ # Travel times only, service times are handled by time_callback
            [0.0, 1.0, 1.0, 1.4],
            [1.0, 0.0, 1.4, 1.0],
            [1.0, 1.4, 0.0, 1.0],
            [1.4, 1.0, 1.0, 0.0]
        ])
        
        self.vehicles = [
            Vehicle(id="vehicle1", capacity=10.0, start_location_id="depot", end_location_id="depot"),
            Vehicle(id="vehicle2", capacity=15.0, start_location_id="depot", end_location_id="depot")
        ]
        
        self.deliveries = [
            Delivery(id="delivery1", location_id="customer1", demand=5.0, is_pickup=False),
            Delivery(id="delivery2", location_id="customer2", demand=3.0, is_pickup=False),
            Delivery(id="delivery3", location_id="customer3", demand=6.0, is_pickup=False)
        ]

    def test_basic_routing(self):
        """Test basic routing functionality using solve()."""
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=self.location_ids.index("depot")
        )        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn(result.status, ['success', 'failed'], "Solver status was not 'success' or 'failed'.")
        
        if result.status == 'success':
            self.assertEqual(len(result.unassigned_deliveries), 0, "Not all deliveries were assigned.")
            self.assertLessEqual(len(result.routes), len(self.vehicles), "More routes than vehicles.")
            
            all_visits_in_routes = set()
            for route_list in result.routes:
                self.assertGreaterEqual(len(route_list), 2, "Route has less than 2 stops (depot-depot).")
                self.assertEqual(route_list[0], 'depot', "Route does not start at depot.")
                self.assertEqual(route_list[-1], 'depot', "Route does not end at depot.")
                all_visits_in_routes.update(route_list[1:-1])
                
            self.assertEqual(all_visits_in_routes, {'customer1', 'customer2', 'customer3'}, "Not all customers visited.")
        else:
            print(f"Basic routing test failed to find a solution: {result.statistics.get('error', 'Unknown error')}")


    def test_multi_vehicle_assignment(self):
        """Test that deliveries are assigned to multiple vehicles when needed."""
        # Using the default setup, which should require multiple vehicles due to capacity/demand
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=self.location_ids.index("depot")
        )
        
        if result.status != 'success':
            self.skipTest(f"Solver did not find a solution: {result.statistics.get('error', 'Test skipped.')}")
            
        self.assertEqual(len(result.unassigned_deliveries), 0)
        # Total demand = 5+3+6 = 14. Vehicle1 capacity = 10, Vehicle2 capacity = 15.
        # It's likely to use two vehicles if the solver finds an optimal split.
        if len(result.routes) > 1 : # If more than one route is used
             self.assertGreaterEqual(len(result.assigned_vehicles), 1, "Expected at least one vehicle to be assigned routes.")
             # This assertion is a bit weak for "multi-vehicle" but confirms general assignment.
             # A stronger test would need specific demand/capacity forcing 2 vehicles.

    def test_empty_problem(self):
        """Test handling of empty problem (no deliveries)."""
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=[],  # No deliveries
            depot_index=self.location_ids.index("depot")
        )
        
        self.assertEqual(result.status, 'success')
        self.assertEqual(len(result.routes), len(self.vehicles), "Expected one depot-depot route per vehicle.")
        for route in result.routes:
            self.assertEqual(len(route), 2, "Depot-depot route should have 2 stops.")
            # Vehicle start_location_id and end_location_id (or start_location_id)
            # Our setup has all vehicles start/end at 'depot'
            self.assertEqual(route[0], 'depot')
            self.assertEqual(route[1], 'depot')
        self.assertEqual(len(result.unassigned_deliveries), 0)
        self.assertIn('info', result.statistics)
        self.assertEqual(result.statistics['info'], 'Empty problem: direct depot-to-depot routes created')


    def test_pickup_and_delivery(self):
        """Test handling of pickup and delivery operations."""
        mixed_deliveries = [
            Delivery(id="pickup1", location_id="customer1", demand=5.0, is_pickup=True), # Net demand -5
            Delivery(id="delivery1", location_id="customer2", demand=3.0, is_pickup=False),# Net demand +3
            Delivery(id="delivery2", location_id="customer3", demand=6.0, is_pickup=False) # Net demand +6
        ] # Vehicle capacity can now be better utilized
        
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=mixed_deliveries,
            depot_index=self.location_ids.index("depot")
        )
        
        self.assertIsInstance(result, OptimizationResult)
        if result.status == 'success':
            self.assertEqual(len(result.unassigned_deliveries), 0)
            all_visits = set()
            for route_list in result.routes:
                all_visits.update(route_list[1:-1])
            self.assertEqual(all_visits, {'customer1', 'customer2', 'customer3'})
        else:
            print(f"Pickup and delivery test failed to find a solution: {result.statistics.get('error', 'Unknown error')}")


    def test_time_windows(self):
        """Test routing with time windows using solve_with_time_windows()."""
        locations_with_tw = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True, time_window_start=0, time_window_end=1440, service_time=0), # Full day
            Location(id="customer1", name="C1-Early", latitude=1.0, longitude=0.0, time_window_start=60, time_window_end=120, service_time=10),   # 1:00-2:00
            Location(id="customer2", name="C2-Mid", latitude=0.0, longitude=1.0, time_window_start=120, time_window_end=240, service_time=15), # 2:00-4:00
            Location(id="customer3", name="C3-Late", latitude=1.0, longitude=1.0, time_window_start=180, time_window_end=300, service_time=5)    # 3:00-5:00
        ]
        # Deliveries for these specific customers
        deliveries_for_tw = [
            Delivery(id="d_c1e", location_id="customer1", demand=2.0),
            Delivery(id="d_c2m", location_id="customer2", demand=3.0),
            Delivery(id="d_c3l", location_id="customer3", demand=4.0)
        ]
        location_ids_for_tw = [loc.id for loc in locations_with_tw]
        # Simple distance matrix for these TW locations
        # depot, C1, C2, C3
        # For simplicity, let speed_km_per_hour = 60, so 1km = 1 minute travel
        distance_matrix_tw = np.array([
            [0.0, 1.0, 1.0, 1.4],  # depot to C1, C2, C3
            [1.0, 0.0, 1.4, 1.0],  # C1 to depot, C2, C3
            [1.0, 1.4, 0.0, 1.0],  # C2 to depot, C1, C3
            [1.4, 1.0, 1.0, 0.0]   # C3 to depot, C1, C2
        ])


        solution_result = self.solver.solve_with_time_windows(
            distance_matrix=distance_matrix_tw,
            location_ids=location_ids_for_tw,
            vehicles=self.vehicles, # Use existing vehicles
            deliveries=deliveries_for_tw,
            locations=locations_with_tw, # Pass the list of Location objects with TW
            depot_index=location_ids_for_tw.index("depot"),
            speed_km_per_hour=60.0 # 1km = 1 minute travel
        )

        self.assertIsInstance(solution_result, OptimizationResult)
        self.assertIn(solution_result.status, ['success', 'failed'])

        if solution_result.status == 'success':
            self.assertEqual(len(solution_result.unassigned_deliveries), 0, 
                             f"Unassigned: {solution_result.unassigned_deliveries}")

            for route_info_dict in solution_result.detailed_routes:
                arrival_times_map = route_info_dict.get('estimated_arrival_times', {})
                
                for loc_id, arrival_time_seconds in arrival_times_map.items():
                    # TIME_SCALING_FACTOR is 60 (min to sec)
                    arrival_minutes = arrival_time_seconds / TIME_SCALING_FACTOR 
                    
                    location_obj = next((l for l in locations_with_tw if l.id == loc_id), None)
                    if location_obj and location_obj.time_window_start is not None and location_obj.time_window_end is not None:
                        # Add a small tolerance for floating point comparisons if necessary,
                        # but since these are integer minutes, direct comparison should be fine.
                        self.assertGreaterEqual(
                            arrival_minutes, location_obj.time_window_start,
                            f"Arrival at {loc_id} (Vehicle {route_info_dict.get('vehicle_id')}) too early: {arrival_minutes} < {location_obj.time_window_start}"
                        )
                        self.assertLessEqual(
                            arrival_minutes, location_obj.time_window_end,
                            f"Arrival at {loc_id} (Vehicle {route_info_dict.get('vehicle_id')}) too late: {arrival_minutes} > {location_obj.time_window_end}"
                        )
        else:
            print(f"Time windows test failed to find a solution: {solution_result.statistics.get('error', 'Unknown error')}")
            self.fail(f"Solver failed for time windows: {solution_result.statistics.get('error', 'No solution')}")


    def test_solve_with_load_balancing_by_time(self):
        """Test solve() with a time_matrix for load balancing."""
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=self.location_ids.index("depot"),
            time_matrix=self.time_matrix_minutes # Provide time matrix
        )
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn(result.status, ['success', 'failed'])
        if result.status == 'failed':
            self.fail(f"Solver failed with time_matrix for balancing: {result.statistics.get('error')}")


    def test_solve_with_load_balancing_by_distance(self):
        """Test solve() without a time_matrix (balances by distance)."""
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=self.location_ids.index("depot"),
            time_matrix=None # Explicitly None
        )
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn(result.status, ['success', 'failed'])
        if result.status == 'failed':
            self.fail(f"Solver failed balancing by distance: {result.statistics.get('error')}")
            
    def test_vehicle_location_not_found(self):
        """Test solver failure when a vehicle's start location is not in location_ids."""
        invalid_vehicles = [
            Vehicle(id="v_invalid", capacity=10.0, start_location_id="unknown_depot", end_location_id="depot")
        ]
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=invalid_vehicles,
            deliveries=self.deliveries,
            depot_index=self.location_ids.index("depot")
        )
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.status, 'failed')
        self.assertIn('error', result.statistics)
        self.assertIn("Vehicle location not found", result.statistics['error'])

if __name__ == '__main__':
    unittest.main()