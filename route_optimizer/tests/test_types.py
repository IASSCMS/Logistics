import unittest
from dataclasses import fields, is_dataclass
from typing import List, Dict, Any, Tuple, Optional

from route_optimizer.core.types_1 import (
    Location,
    OptimizationResult,
    RouteSegment,
    DetailedRoute,
    ReroutingInfo,
    validate_optimization_result
)

class TestLocationDataclass(unittest.TestCase):
    def test_location_creation_all_fields(self):
        loc = Location(
            id="loc1",
            latitude=34.0522,
            longitude=-118.2437,
            name="Downtown LA",
            address="123 Main St, Los Angeles, CA",
            is_depot=False,
            time_window_start=540, # 9 AM
            time_window_end=1020,  # 5 PM
            service_time=20
        )
        self.assertEqual(loc.id, "loc1")
        self.assertEqual(loc.latitude, 34.0522)
        self.assertEqual(loc.longitude, -118.2437)
        self.assertEqual(loc.name, "Downtown LA")
        self.assertEqual(loc.address, "123 Main St, Los Angeles, CA")
        self.assertFalse(loc.is_depot)
        self.assertEqual(loc.time_window_start, 540)
        self.assertEqual(loc.time_window_end, 1020)
        self.assertEqual(loc.service_time, 20)

    def test_location_creation_required_fields_and_defaults(self):
        loc = Location(id="loc2", latitude=40.7128, longitude=-74.0060)
        self.assertEqual(loc.id, "loc2")
        self.assertEqual(loc.latitude, 40.7128)
        self.assertEqual(loc.longitude, -74.0060)
        self.assertIsNone(loc.name)
        self.assertIsNone(loc.address)
        self.assertFalse(loc.is_depot) # Default
        self.assertIsNone(loc.time_window_start) # Default
        self.assertIsNone(loc.time_window_end) # Default
        self.assertEqual(loc.service_time, 15) # Default

    def test_location_post_init_type_conversion(self):
        loc_str_coords = Location(id="loc3", latitude="35.6895", longitude="139.6917")
        self.assertIsInstance(loc_str_coords.latitude, float)
        self.assertIsInstance(loc_str_coords.longitude, float)
        self.assertEqual(loc_str_coords.latitude, 35.6895)
        self.assertEqual(loc_str_coords.longitude, 139.6917)

        loc_float_coords = Location(id="loc4", latitude=35.6895, longitude=139.6917)
        self.assertIsInstance(loc_float_coords.latitude, float)
        self.assertIsInstance(loc_float_coords.longitude, float)

    def test_location_is_dataclass(self):
        self.assertTrue(is_dataclass(Location))

class TestOptimizationResultDataclass(unittest.TestCase):
    def test_optimization_result_creation_required_fields_and_defaults(self):
        res = OptimizationResult(status="success")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.routes, [])
        self.assertEqual(res.total_distance, 0.0)
        self.assertEqual(res.total_cost, 0.0)
        self.assertEqual(res.assigned_vehicles, {})
        self.assertEqual(res.unassigned_deliveries, [])
        self.assertEqual(res.detailed_routes, [])
        self.assertEqual(res.statistics, {})

    def test_optimization_result_from_dict_valid(self):
        data = {
            "status": "success",
            "routes": [["L1", "L2"]],
            "total_distance": 100.5,
            "total_cost": 50.25,
            "assigned_vehicles": {"V1": 0},
            "unassigned_deliveries": ["D1"],
            "detailed_routes": [{"vehicle_id": "V1", "stops": ["L1", "L2"]}],
            "statistics": {"time": 120}
        }
        res = OptimizationResult.from_dict(data)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.routes, [["L1", "L2"]])
        self.assertEqual(res.total_distance, 100.5)
        self.assertEqual(res.total_cost, 50.25)
        self.assertEqual(res.assigned_vehicles, {"V1": 0})
        self.assertEqual(res.unassigned_deliveries, ["D1"])
        self.assertEqual(res.detailed_routes, [{"vehicle_id": "V1", "stops": ["L1", "L2"]}])
        self.assertEqual(res.statistics, {"time": 120})

    def test_optimization_result_from_dict_missing_fields(self):
        data = {"status": "failed"}
        res = OptimizationResult.from_dict(data)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.routes, []) # Default
        self.assertEqual(res.total_distance, 0.0) # Default
        # ... and so on for other default fields

    def test_optimization_result_from_dict_none_input(self):
        res = OptimizationResult.from_dict(None)
        self.assertEqual(res.status, "error")
        self.assertIn("Input data for OptimizationResult was None", res.statistics.get("error", ""))

    def test_optimization_result_from_dict_conversion_error(self):
        # Test with a non-dict input that might cause an error in .get()
        # For example, if data was a list instead of a dict
        with self.assertLogs(logger='route_optimizer.core.types_1', level='ERROR') as cm:
            res = OptimizationResult.from_dict(["not_a_dict"])
        self.assertEqual(res.status, "error")
        self.assertIn("Conversion error from dict", res.statistics.get("error", ""))
        self.assertTrue(any("AttributeError" in log_msg for log_msg in cm.output) or \
                        any("TypeError" in log_msg for log_msg in cm.output))


    def test_optimization_result_is_dataclass(self):
        self.assertTrue(is_dataclass(OptimizationResult))

class TestRouteSegmentDataclass(unittest.TestCase):
    def test_route_segment_creation(self):
        seg = RouteSegment(
            from_location="A",
            to_location="B",
            path=["A", "Inter", "B"],
            distance=10.5,
            estimated_time=15.0
        )
        self.assertEqual(seg.from_location, "A")
        self.assertEqual(seg.to_location, "B")
        self.assertEqual(seg.path, ["A", "Inter", "B"])
        self.assertEqual(seg.distance, 10.5)
        self.assertEqual(seg.estimated_time, 15.0)

    def test_route_segment_optional_time(self):
        seg = RouteSegment(from_location="C", to_location="D", path=["C", "D"], distance=5.0)
        self.assertIsNone(seg.estimated_time)

    def test_route_segment_is_dataclass(self):
        self.assertTrue(is_dataclass(RouteSegment))

class TestDetailedRouteDataclass(unittest.TestCase):
    def test_detailed_route_creation(self):
        seg1 = RouteSegment(from_location="A", to_location="B", path=["A","B"], distance=5)
        route = DetailedRoute(
            vehicle_id="V1",
            stops=["A", "B", "C"],
            segments=[seg1],
            total_distance=10.0,
            total_time=20.0,
            capacity_utilization=0.75,
            estimated_arrival_times={"B": 10, "C": 20}
        )
        self.assertEqual(route.vehicle_id, "V1")
        self.assertEqual(route.stops, ["A", "B", "C"])
        self.assertEqual(len(route.segments), 1)
        self.assertEqual(route.segments[0].from_location, "A")
        self.assertEqual(route.total_distance, 10.0)
        self.assertEqual(route.total_time, 20.0)
        self.assertEqual(route.capacity_utilization, 0.75)
        self.assertEqual(route.estimated_arrival_times, {"B": 10, "C": 20})

    def test_detailed_route_defaults(self):
        route = DetailedRoute(vehicle_id="V2")
        self.assertEqual(route.vehicle_id, "V2")
        self.assertEqual(route.stops, [])
        self.assertEqual(route.segments, [])
        self.assertEqual(route.total_distance, 0.0)
        self.assertEqual(route.total_time, 0.0)
        self.assertEqual(route.capacity_utilization, 0.0)
        self.assertEqual(route.estimated_arrival_times, {})

    def test_detailed_route_is_dataclass(self):
        self.assertTrue(is_dataclass(DetailedRoute))

class TestReroutingInfoDataclass(unittest.TestCase):
    def test_rerouting_info_creation(self):
        info = ReroutingInfo(
            reason="traffic",
            traffic_factors=3,
            completed_deliveries=5,
            remaining_deliveries=10,
            delay_locations=["locX", "locY"],
            blocked_segments=[("A", "B"), ("C", "D")]
        )
        self.assertEqual(info.reason, "traffic")
        self.assertEqual(info.traffic_factors, 3)
        self.assertEqual(info.completed_deliveries, 5)
        self.assertEqual(info.remaining_deliveries, 10)
        self.assertEqual(info.delay_locations, ["locX", "locY"])
        self.assertEqual(info.blocked_segments, [("A", "B"), ("C", "D")])

    def test_rerouting_info_defaults(self):
        info = ReroutingInfo(reason="roadblock")
        self.assertEqual(info.reason, "roadblock")
        self.assertEqual(info.traffic_factors, 0)
        self.assertEqual(info.completed_deliveries, 0)
        self.assertEqual(info.remaining_deliveries, 0)
        self.assertEqual(info.delay_locations, [])
        self.assertEqual(info.blocked_segments, [])

    def test_rerouting_info_is_dataclass(self):
        self.assertTrue(is_dataclass(ReroutingInfo))

class TestValidateOptimizationResult(unittest.TestCase):
    def setUp(self):
        self.base_success_result = {
            "status": "success",
            "routes": [["depot", "loc1", "depot"]],
            "total_distance": 10.0,
            "total_cost": 5.0,
            "assigned_vehicles": {"V1": 0},
            "unassigned_deliveries": [],
            "detailed_routes": [
                {
                    "vehicle_id": "V1",
                    "stops": ["depot", "loc1", "depot"],
                    "segments": [
                        {"from": "depot", "to": "loc1", "distance": 5.0, "path": []},
                        {"from": "loc1", "to": "depot", "distance": 5.0, "path": []}
                    ],
                    "total_distance": 10.0,
                    "total_time": 0.5,
                    "capacity_utilization": 0.1,
                    "estimated_arrival_times": {"loc1": 10}
                }
            ],
            "statistics": {"computation_time_ms": 100}
        }
        self.base_failed_result = {"status": "failed"}

    def test_valid_success_result_full(self):
        self.assertTrue(validate_optimization_result(self.base_success_result))

    def test_valid_success_result_minimal_routes(self):
        result = {"status": "success", "routes": [["A", "B"]]}
        self.assertTrue(validate_optimization_result(result))

    def test_valid_failed_result(self):
        self.assertTrue(validate_optimization_result(self.base_failed_result))

    def test_invalid_missing_status(self):
        with self.assertRaisesRegex(ValueError, "Missing required field: status"):
            validate_optimization_result({})

    def test_invalid_status_value(self):
        with self.assertRaisesRegex(ValueError, "Invalid status value: pending"):
            validate_optimization_result({"status": "pending"})

    def test_invalid_success_missing_routes(self):
        with self.assertRaisesRegex(ValueError, "Missing 'routes' in successful result"):
            validate_optimization_result({"status": "success"})

    def test_invalid_routes_not_list(self):
        with self.assertRaisesRegex(ValueError, "'routes' must be a list"):
            validate_optimization_result({"status": "success", "routes": "not_a_list"})

    def test_invalid_assigned_vehicles_not_dict(self):
        result = {**self.base_success_result, "assigned_vehicles": "not_a_dict"}
        with self.assertRaisesRegex(ValueError, "'assigned_vehicles' must be a dictionary"):
            validate_optimization_result(result)

    def test_invalid_assigned_vehicles_bad_index_type(self):
        result = {**self.base_success_result, "assigned_vehicles": {"V1": "zero"}}
        with self.assertRaisesRegex(ValueError, "Invalid route index zero for vehicle V1"):
            validate_optimization_result(result)
            
    def test_invalid_assigned_vehicles_bad_index_out_of_bounds(self):
        result = {**self.base_success_result, "assigned_vehicles": {"V1": 1}} # Only 1 route at index 0
        with self.assertRaisesRegex(ValueError, "Invalid route index 1 for vehicle V1"):
            validate_optimization_result(result)

    def test_invalid_detailed_routes_not_list(self):
        result = {**self.base_success_result, "detailed_routes": "not_a_list"}
        with self.assertRaisesRegex(ValueError, "'detailed_routes' must be a list"):
            validate_optimization_result(result)

    def test_invalid_detailed_route_item_not_dict(self):
        result = {**self.base_success_result, "detailed_routes": ["not_a_dict"]}
        with self.assertRaisesRegex(ValueError, "Route at index 0 must be a dictionary"):
            validate_optimization_result(result)

    def test_invalid_detailed_route_missing_vehicle_id(self):
        result = {**self.base_success_result, "detailed_routes": [{}]}
        with self.assertRaisesRegex(ValueError, "Missing 'vehicle_id' in route at index 0"):
            validate_optimization_result(result)

    def test_invalid_detailed_route_missing_stops_and_segments(self):
        result = {**self.base_success_result, "detailed_routes": [{"vehicle_id": "V1"}]}
        with self.assertRaisesRegex(ValueError, "Route at index 0 must have either 'stops' or 'segments'"):
            validate_optimization_result(result)

    def test_invalid_detailed_route_segment_not_dict(self):
        detailed_route = {**self.base_success_result["detailed_routes"][0], "segments": ["not_a_dict"]}
        result = {**self.base_success_result, "detailed_routes": [detailed_route]}
        with self.assertRaisesRegex(ValueError, "Segment at index 0 in route 0 must be a dictionary"):
            validate_optimization_result(result)

    def test_invalid_detailed_route_segment_missing_field(self):
        segment_missing_from = {"to": "L2", "distance": 5.0}
        detailed_route = {**self.base_success_result["detailed_routes"][0], "segments": [segment_missing_from]}
        result = {**self.base_success_result, "detailed_routes": [detailed_route]}
        with self.assertRaisesRegex(ValueError, "Missing 'from' in segment 0 of route 0"):
            validate_optimization_result(result)
            
        segment_missing_to = {"from": "L1", "distance": 5.0}
        detailed_route_2 = {**self.base_success_result["detailed_routes"][0], "segments": [segment_missing_to]}
        result_2 = {**self.base_success_result, "detailed_routes": [detailed_route_2]}
        with self.assertRaisesRegex(ValueError, "Missing 'to' in segment 0 of route 0"):
            validate_optimization_result(result_2)

        segment_missing_distance = {"from": "L1", "to": "L2"}
        detailed_route_3 = {**self.base_success_result["detailed_routes"][0], "segments": [segment_missing_distance]}
        result_3 = {**self.base_success_result, "detailed_routes": [detailed_route_3]}
        with self.assertRaisesRegex(ValueError, "Missing 'distance' in segment 0 of route 0"):
            validate_optimization_result(result_3)


if __name__ == '__main__':
    unittest.main()
