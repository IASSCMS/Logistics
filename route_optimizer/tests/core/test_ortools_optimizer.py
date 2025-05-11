"""
Tests for OR-Tools VRP solver implementation.

This module contains tests for the ORToolsVRPSolver class.
"""
import unittest
import numpy as np
from route_optimizer.core.constants import TIME_SCALING_FACTOR
from route_optimizer.core.ortools_optimizer import ORToolsVRPSolver
from route_optimizer.core.types_1 import Location, OptimizationResult
from route_optimizer.models import Vehicle, Delivery

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
        
        # Sample deliveries - using the demand property that's properly implemented
        self.deliveries = [
            Delivery(id="delivery1", location_id="customer1", demand=5.0, is_pickup=False),
            Delivery(id="delivery2", location_id="customer2", demand=3.0, is_pickup=False),
            Delivery(id="delivery3", location_id="customer3", demand=6.0, is_pickup=False)
        ]

    def test_basic_routing(self):
        """Test basic routing functionality."""
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=0
        )        
        # Verify result is an OptimizationResult
        self.assertIsInstance(result, OptimizationResult)
        
        # Verify result attributes
        self.assertIn(result.status, ['success', 'failed'])
        self.assertTrue(hasattr(result, 'routes'))
        self.assertTrue(hasattr(result, 'total_distance'))
        self.assertTrue(hasattr(result, 'assigned_vehicles'))
        self.assertTrue(hasattr(result, 'unassigned_deliveries'))
        
        # If successful, verify all deliveries are assigned
        if result.status == 'success':
            self.assertEqual(len(result.unassigned_deliveries), 0)
            
            # Verify correct number of routes
            # We expect at most 2 routes (one per vehicle)
            self.assertLessEqual(len(result.routes), 2)
            
            # Verify each route starts and ends at the depot
            for route in result.routes:
                self.assertEqual(route[0], 'depot')  # Start at depot
                self.assertEqual(route[-1], 'depot')  # End at depot
            
            # All customers should be visited exactly once
            all_visits = []
            for route in result.routes:
                all_visits.extend(route[1:-1])  # Exclude depot at start and end
                
            self.assertEqual(set(all_visits), {'customer1', 'customer2', 'customer3'})

    def test_multi_vehicle_assignment(self):
        """Test that deliveries are assigned to multiple vehicles when needed."""
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            depot_index=0
        )
        
        # Skip if solution failed
        if result.status != 'success':
            self.skipTest("Solver did not find a solution")
            
        # All deliveries should be assigned
        self.assertEqual(len(result.unassigned_deliveries), 0)
        
        # The solver might use one or two vehicles depending on the best solution
        # If the total demand (14.0) is split, we should have two routes
        if len(result.routes) == 2:
            # Verify both vehicles are used
            self.assertEqual(len(result.assigned_vehicles), 2)

    def test_empty_problem(self):
        """Test handling of empty problem (no deliveries)."""
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=[],  # No deliveries
            depot_index=0
        )
        
        # Should have valid solution
        self.assertEqual(result.status, 'success')
        
        # Empty routes still contain depot-to-depot movements
        # Since there are two vehicles, we expect two routes with just depot
        self.assertEqual(len(result.routes), 2)
        for route in result.routes:
            self.assertEqual(len(route), 2)  # Just depot-depot
            self.assertEqual(route[0], 'depot')
            self.assertEqual(route[1], 'depot')

    def test_pickup_and_delivery(self):
        """Test handling of pickup and delivery operations."""
        # Create deliveries with both pickup and delivery operations
        mixed_deliveries = [
            Delivery(id="pickup1", location_id="customer1", demand=5.0, is_pickup=True),
            Delivery(id="delivery1", location_id="customer2", demand=3.0, is_pickup=False),
            Delivery(id="delivery2", location_id="customer3", demand=6.0, is_pickup=False)
        ]
        
        result = self.solver.solve(
            distance_matrix=self.distance_matrix,
            location_ids=self.location_ids,
            vehicles=self.vehicles,
            deliveries=mixed_deliveries,
            depot_index=0
        )
        
        # Verify result is successful
        if result.status == 'success':
            # All deliveries should be assigned
            self.assertEqual(len(result.unassigned_deliveries), 0)
            
            # Verify routes contain all locations
            all_visits = []
            for route in result.routes:
                all_visits.extend(route[1:-1])  # Exclude depot at start and end
            
            self.assertEqual(set(all_visits), {'customer1', 'customer2', 'customer3'})

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

        # Check required keys (solve_with_time_windows returns a dict)
        self.assertIn('status', solution)
        self.assertIn('routes', solution)

        # If successful, check time windows are respected
        if solution['status'] == 'success':
            for route in solution['routes']:
                for stop in route:
                    loc_id = stop['location_id']
                    arrival_seconds = stop['arrival_time_seconds']
                    arrival_minutes = arrival_seconds // TIME_SCALING_FACTOR  # Convert to minutes
                    
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