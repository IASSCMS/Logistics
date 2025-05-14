import unittest
from unittest.mock import patch, MagicMock, ANY, call
import copy
import numpy as np

from route_optimizer.services.rerouting_service import ReroutingService
from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.core.types_1 import Location, OptimizationResult, ReroutingInfo
from route_optimizer.models import Vehicle, Delivery # Assuming these are dataclasses

class TestReroutingServiceInitialization(unittest.TestCase):
    def test_init_with_default_optimization_service(self):
        service = ReroutingService()
        self.assertIsInstance(service.optimization_service, OptimizationService)

    def test_init_with_provided_optimization_service(self):
        mock_opt_service = MagicMock(spec=OptimizationService)
        service = ReroutingService(optimization_service=mock_opt_service)
        self.assertEqual(service.optimization_service, mock_opt_service)


class TestReroutingServiceHelperMethods(unittest.TestCase):
    def setUp(self):
        self.service = ReroutingService()
        self.original_deliveries = [
            Delivery(id="d1", location_id="loc_A", demand=10),
            Delivery(id="d2", location_id="loc_B", demand=5),
            Delivery(id="d3", location_id="loc_C", demand=8),
        ]
        self.vehicles = [
            Vehicle(id="v1", capacity=100, start_location_id="depot", end_location_id="depot"),
            Vehicle(id="v2", capacity=100, start_location_id="depot", end_location_id="depot"),
        ]
        self.locations = [
            Location(id="depot", latitude=0, longitude=0, is_depot=True),
            Location(id="loc_A", latitude=1, longitude=1),
            Location(id="loc_B", latitude=2, longitude=2),
            Location(id="loc_C", latitude=3, longitude=3),
            Location(id="loc_D", latitude=4, longitude=4), # Next stop after L2
        ]

    def test_get_remaining_deliveries(self):
        completed_ids = ["d1"]
        remaining = self.service._get_remaining_deliveries(self.original_deliveries, completed_ids)
        self.assertEqual(len(remaining), 2)
        self.assertTrue(any(d.id == "d2" for d in remaining))
        self.assertTrue(any(d.id == "d3" for d in remaining))
        self.assertFalse(any(d.id == "d1" for d in remaining))

    def test_get_remaining_deliveries_none_completed(self):
        remaining = self.service._get_remaining_deliveries(self.original_deliveries, [])
        self.assertEqual(len(remaining), 3)

    def test_get_remaining_deliveries_all_completed(self):
        completed_ids = ["d1", "d2", "d3"]
        remaining = self.service._get_remaining_deliveries(self.original_deliveries, completed_ids)
        self.assertEqual(len(remaining), 0)

    def test_get_remaining_deliveries_empty_original(self):
        remaining = self.service._get_remaining_deliveries([], ["d1"])
        self.assertEqual(len(remaining), 0)
        with self.assertLogs(logger='route_optimizer.services.rerouting_service', level='WARNING') as cm:
            self.service._get_remaining_deliveries([], ["d1"])
            self.assertIn("Original deliveries list is empty, but completed IDs were provided", cm.output[0])
        
        with self.assertLogs(logger='route_optimizer.services.rerouting_service', level='INFO') as cm:
            self.service._get_remaining_deliveries([], [])
            self.assertIn("Original deliveries list is empty", cm.output[0])


    def test_update_vehicle_positions_moves_to_next_stop(self):
        current_routes_dto = OptimizationResult(
            status="success",
            assigned_vehicles={"v1": 0},
            detailed_routes=[
                {"vehicle_id": "v1", "stops": ["depot", "loc_A", "loc_B", "depot"]}
            ]
        )
        # Delivery d1 is at loc_A
        updated_vehicles = self.service._update_vehicle_positions(
            self.vehicles, current_routes_dto, ["d1"], self.original_deliveries
        )
        v1_updated = next(v for v in updated_vehicles if v.id == "v1")
        self.assertEqual(v1_updated.start_location_id, "loc_B")

    def test_update_vehicle_positions_last_stop_completed(self):
        current_routes_dto = OptimizationResult(
            status="success",
            assigned_vehicles={"v1": 0},
            detailed_routes=[
                {"vehicle_id": "v1", "stops": ["depot", "loc_A", "loc_B"]} # Ends at loc_B
            ]
        )
        # Delivery d2 is at loc_B
        updated_vehicles = self.service._update_vehicle_positions(
            self.vehicles, current_routes_dto, ["d2"], self.original_deliveries
        )
        v1_updated = next(v for v in updated_vehicles if v.id == "v1")
        self.assertEqual(v1_updated.start_location_id, "loc_B") # Stays at last completed stop


    def test_update_vehicle_positions_no_completed_deliveries_on_route(self):
        v1_original_start = self.vehicles[0].start_location_id
        current_routes_dto = OptimizationResult(
            status="success",
            assigned_vehicles={"v1": 0},
            detailed_routes=[
                {"vehicle_id": "v1", "stops": ["depot", "loc_C", "depot"]}
            ]
        )
        # No deliveries associated with loc_C are in completed_delivery_ids
        updated_vehicles = self.service._update_vehicle_positions(
            self.vehicles, current_routes_dto, ["d1"], self.original_deliveries # d1 is at loc_A
        )
        v1_updated = next(v for v in updated_vehicles if v.id == "v1")
        self.assertEqual(v1_updated.start_location_id, v1_original_start)


    def test_update_vehicle_positions_vehicle_not_in_plan(self):
        v2_original_start = self.vehicles[1].start_location_id
        current_routes_dto = OptimizationResult(
            status="success",
            assigned_vehicles={"v1": 0}, # v2 not assigned
            detailed_routes=[
                {"vehicle_id": "v1", "stops": ["depot", "loc_A", "depot"]}
            ]
        )
        updated_vehicles = self.service._update_vehicle_positions(
            self.vehicles, current_routes_dto, ["d1"], self.original_deliveries
        )
        v2_updated = next(v for v in updated_vehicles if v.id == "v2")
        self.assertEqual(v2_updated.start_location_id, v2_original_start)

    def test_update_vehicle_positions_empty_detailed_routes(self):
        v1_original_start = self.vehicles[0].start_location_id
        current_routes_dto = OptimizationResult(
            status="success",
            assigned_vehicles={"v1": 0},
            detailed_routes=[] # Empty detailed_routes
        )
        updated_vehicles = self.service._update_vehicle_positions(
            self.vehicles, current_routes_dto, ["d1"], self.original_deliveries
        )
        v1_updated = next(v for v in updated_vehicles if v.id == "v1")
        self.assertEqual(v1_updated.start_location_id, v1_original_start)
        
    def test_update_vehicle_positions_route_has_no_stops(self):
        v1_original_start = self.vehicles[0].start_location_id
        current_routes_dto = OptimizationResult(
            status="success",
            assigned_vehicles={"v1": 0},
            detailed_routes=[{"vehicle_id": "v1", "stops": []}] # Route with no stops
        )
        updated_vehicles = self.service._update_vehicle_positions(
            self.vehicles, current_routes_dto, ["d1"], self.original_deliveries
        )
        v1_updated = next(v for v in updated_vehicles if v.id == "v1")
        self.assertEqual(v1_updated.start_location_id, v1_original_start)


class TestReroutingServiceMainMethods(unittest.TestCase):
    def setUp(self):
        self.mock_opt_service = MagicMock(spec=OptimizationService)
        self.service = ReroutingService(optimization_service=self.mock_opt_service)

        self.locations = [
            Location(id="L0", latitude=0, longitude=0, is_depot=True, service_time=0),
            Location(id="L1", latitude=1, longitude=0, service_time=10),
            Location(id="L2", latitude=2, longitude=0, service_time=10),
        ]
        self.vehicles = [Vehicle(id="V1", capacity=10, start_location_id="L0")]
        self.original_deliveries = [
            Delivery(id="D1", location_id="L1", demand=5),
            Delivery(id="D2", location_id="L2", demand=5),
        ]
        self.current_routes_dto = OptimizationResult(
            status="success",
            routes=[["L0", "L1", "L2", "L0"]],
            assigned_vehicles={"V1": 0},
            detailed_routes=[{"vehicle_id": "V1", "stops": ["L0", "L1", "L2", "L0"]}],
            statistics={}
        )
        self.mock_new_routes_dto = OptimizationResult(
            status="success", 
            total_distance=10, 
            statistics={}
        )
        self.mock_opt_service.optimize_routes.return_value = self.mock_new_routes_dto

    @patch('route_optimizer.services.rerouting_service.ReroutingService._get_remaining_deliveries')
    @patch('route_optimizer.services.rerouting_service.ReroutingService._update_vehicle_positions')
    def test_reroute_for_traffic_success(self, mock_update_pos, mock_get_remaining):
        mock_get_remaining.return_value = self.original_deliveries # Simplified
        mock_update_pos.return_value = self.vehicles # Simplified
        traffic_data = {(0, 1): 1.5}
        
        result = self.service.reroute_for_traffic(
            self.current_routes_dto, self.locations, self.vehicles,
            self.original_deliveries, ["some_completed"], traffic_data
        )

        mock_get_remaining.assert_called_once_with(self.original_deliveries, ["some_completed"])
        mock_update_pos.assert_called_once_with(self.vehicles, self.current_routes_dto, ["some_completed"], self.original_deliveries)
        self.mock_opt_service.optimize_routes.assert_called_once_with(
            locations=self.locations,
            vehicles=self.vehicles, # Result from mock_update_pos
            deliveries=self.original_deliveries, # Result from mock_get_remaining
            consider_traffic=True,
            traffic_data=traffic_data
        )
        self.assertEqual(result.status, "success")
        self.assertIn("rerouting_info", result.statistics)
        rerouting_info = result.statistics["rerouting_info"]
        self.assertEqual(rerouting_info["reason"], "traffic")
        self.assertEqual(rerouting_info["traffic_factors"], len(traffic_data))
        self.assertEqual(rerouting_info["completed_deliveries"], 1)

    def test_reroute_for_traffic_exception_in_optimize(self):
        self.mock_opt_service.optimize_routes.side_effect = Exception("Optimize failed")
        result = self.service.reroute_for_traffic(
            self.current_routes_dto, self.locations, self.vehicles,
            self.original_deliveries, [], {}
        )
        self.assertEqual(result.status, "error")
        self.assertIn("Rerouting for traffic failed: Optimize failed", result.statistics["error"])

    @patch('route_optimizer.services.rerouting_service.ReroutingService._get_remaining_deliveries')
    @patch('route_optimizer.services.rerouting_service.ReroutingService._update_vehicle_positions')
    def test_reroute_for_delay_success(self, mock_update_pos, mock_get_remaining):
        mock_get_remaining.return_value = self.original_deliveries
        mock_update_pos.return_value = self.vehicles # This is the mock for _update_vehicle_positions
        
        delayed_location_ids = ["L1"]
        delay_minutes = {"L1": 30}

        # Create a deepcopy for locations to be modified by the service
        locs_for_test = copy.deepcopy(self.locations)
        
        # Get the original service time for L1 *before* the call
        original_l1_service_time = next(loc.service_time for loc in self.locations if loc.id == "L1")

        # Call the method under test
        result = self.service.reroute_for_delay(
            self.current_routes_dto, 
            locs_for_test, # This list will be deepcopied and modified inside reroute_for_delay
            self.vehicles,
            self.original_deliveries, 
            [], # completed_deliveries
            delayed_location_ids, 
            delay_minutes
        )

        # Assertions:
        
        # 1. Verify that _get_remaining_deliveries and _update_vehicle_positions were called
        mock_get_remaining.assert_called_once_with(self.original_deliveries, [])
        mock_update_pos.assert_called_once_with(self.vehicles, self.current_routes_dto, [], self.original_deliveries)

        # 2. Verify that optimization_service.optimize_routes was called with correct parameters
        #    and inspect the 'locations' argument passed to it.
        self.mock_opt_service.optimize_routes.assert_called_once_with(
            locations=ANY, # Use ANY because it's a deepcopy and hard to compare directly
            vehicles=self.vehicles, # This was returned by mock_update_pos
            deliveries=self.original_deliveries, # This was returned by mock_get_remaining
            consider_time_windows=True
        )
        
        # Retrieve the 'locations' list that was actually passed to optimize_routes
        # call_args is a tuple: (positional_args, keyword_args)
        # optimize_routes is called with keyword arguments here.
        passed_kwargs_to_optimize = self.mock_opt_service.optimize_routes.call_args[1]
        updated_locations_arg = passed_kwargs_to_optimize['locations']
        
        # Find the L1 location in the list passed to optimize_routes
        updated_l1_from_opt_call = next((loc for loc in updated_locations_arg if loc.id == "L1"), None)
        self.assertIsNotNone(updated_l1_from_opt_call, "Location L1 not found in arguments to optimize_routes")
        
        # Check if its service_time was updated correctly
        self.assertEqual(updated_l1_from_opt_call.service_time, original_l1_service_time + 30)

        # 3. Verify the ReroutingInfo in the result
        self.assertEqual(result.status, "success")
        self.assertIn("rerouting_info", result.statistics)
        rerouting_info = result.statistics["rerouting_info"]
        self.assertEqual(rerouting_info["reason"], "service_delay")
        self.assertEqual(rerouting_info["delay_locations"], delayed_location_ids)
        self.assertEqual(rerouting_info["completed_deliveries"], 0)
        self.assertEqual(rerouting_info["remaining_deliveries"], len(self.original_deliveries))

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.services.rerouting_service.ReroutingService._get_remaining_deliveries')
    @patch('route_optimizer.services.rerouting_service.ReroutingService._update_vehicle_positions')
    def test_reroute_for_roadblock_success(self, mock_update_pos, mock_get_remaining, mock_create_matrix):
        mock_get_remaining.return_value = self.original_deliveries
        mock_update_pos.return_value = self.vehicles
        
        # Setup mock for DistanceMatrixBuilder.create_distance_matrix
        # It returns (distance_matrix, time_matrix, location_ids)
        # L0, L1, L2
        mock_dist_matrix = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]], dtype=float)
        mock_loc_ids = [loc.id for loc in self.locations]
        mock_create_matrix.return_value = (mock_dist_matrix, None, mock_loc_ids)
        
        blocked_segments = [("L0", "L1")] # Block between L0 (idx 0) and L1 (idx 1)
        
        result = self.service.reroute_for_roadblock(
            self.current_routes_dto, self.locations, self.vehicles,
            self.original_deliveries, [], blocked_segments
        )

        mock_create_matrix.assert_called_once_with(self.locations, use_haversine=True, average_speed_kmh=None)
        
        # Verify optimize_routes was called with traffic_data reflecting the roadblock
        args, kwargs = self.mock_opt_service.optimize_routes.call_args
        expected_traffic_for_roadblocks = {(0, 1): float('inf'), (1, 0): float('inf')}
        self.assertEqual(kwargs['traffic_data'], expected_traffic_for_roadblocks)
        
        self.assertEqual(result.status, "success")
        rerouting_info = result.statistics["rerouting_info"]
        self.assertEqual(rerouting_info["reason"], "roadblock")
        self.assertEqual(rerouting_info["blocked_segments"], blocked_segments)
        self.assertEqual(rerouting_info["blocked_segments_count"], len(blocked_segments))

    def test_reroute_for_roadblock_key_error_in_segment(self):
        # Test when a location ID in blocked_segments is not in self.locations
        mock_dist_matrix = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]], dtype=float)
        mock_loc_ids = [loc.id for loc in self.locations] # L0, L1, L2
        
        with patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix', return_value=(mock_dist_matrix, None, mock_loc_ids)):
            with self.assertLogs(logger='route_optimizer.services.rerouting_service', level='WARNING') as cm:
                self.service.reroute_for_roadblock(
                    self.current_routes_dto, self.locations, self.vehicles,
                    self.original_deliveries, [], [("L0", "UNKNOWN_LOC")]
                )
                self.assertIn("Location ID not found when applying roadblock: L0 or UNKNOWN_LOC", cm.output[0])
        # The rest of the method should still proceed, potentially with an empty traffic_data_for_roadblocks

if __name__ == '__main__':
    unittest.main()

