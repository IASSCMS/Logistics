import dataclasses
from django.test import TestCase
from rest_framework import serializers
from unittest.mock import patch

from route_optimizer.core.constants import DEFAULT_DELIVERY_PRIORITY
from route_optimizer.core.types_1 import OptimizationResult # For OptimizationResultSerializer test
from route_optimizer.api.serializers import (
    LocationSerializer,
    VehicleSerializer,
    DeliverySerializer,
    RouteOptimizationRequestSerializer,
    RouteSegmentSerializer,
    VehicleRouteSerializer,
    ReroutingInfoSerializer,
    StatisticsSerializer,
    OptimizationResultSerializer,
    RouteOptimizationResponseSerializer,
    TrafficDataSerializer,
    ReroutingRequestSerializer
)

class LocationSerializerTests(TestCase):
    def test_valid_location(self):
        data = {
            "id": "loc1", "name": "Location 1", "latitude": 34.0522, "longitude": -118.2437,
            "time_window_start": 540, "time_window_end": 1020, "service_time": 30
        }
        serializer = LocationSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['service_time'], 30) # Check non-default

    def test_location_missing_required_fields(self):
        data = {"id": "loc1", "name": "Location 1"} # Missing latitude, longitude
        serializer = LocationSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('latitude', serializer.errors)
        self.assertIn('longitude', serializer.errors)

    def test_location_default_service_time(self):
        data = {"id": "loc1", "name": "Location 1", "latitude": 34.0522, "longitude": -118.2437}
        serializer = LocationSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['service_time'], 15) # Default value

    def test_location_optional_fields(self):
        data = {
            "id": "loc1", "name": "Location 1", "latitude": 34.0522, "longitude": -118.2437,
            "address": "123 Main St"
        }
        serializer = LocationSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['address'], "123 Main St")
        self.assertIsNone(serializer.validated_data.get('time_window_start'))


class VehicleSerializerTests(TestCase):
    def test_valid_vehicle(self):
        data = {
            "id": "veh1", "capacity": 100.0, "start_location_id": "depot",
            "end_location_id": "depot_end", "cost_per_km": 1.5, "fixed_cost": 50.0,
            "skills": ["refrigeration"]
        }
        serializer = VehicleSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['cost_per_km'], 1.5)

    def test_vehicle_missing_required_fields(self):
        data = {"id": "veh1"} # Missing capacity, start_location_id
        serializer = VehicleSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('capacity', serializer.errors)
        self.assertIn('start_location_id', serializer.errors)

    def test_vehicle_default_values(self):
        data = {"id": "veh1", "capacity": 100.0, "start_location_id": "depot"}
        serializer = VehicleSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['cost_per_km'], 1.0)
        self.assertEqual(serializer.validated_data['fixed_cost'], 0.0)
        self.assertTrue(serializer.validated_data['available'])
        self.assertEqual(serializer.validated_data['skills'], [])


class DeliverySerializerTests(TestCase):
    def test_valid_delivery(self):
        data = {
            "id": "del1", "location_id": "cust1", "demand": 10.0, "priority": 2,
            "required_skills": ["fragile"], "is_pickup": True
        }
        serializer = DeliverySerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['priority'], 2)

    def test_delivery_missing_required_fields(self):
        data = {"id": "del1"} # Missing location_id, demand
        serializer = DeliverySerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('location_id', serializer.errors)
        self.assertIn('demand', serializer.errors)

    def test_delivery_default_values(self):
        data = {"id": "del1", "location_id": "cust1", "demand": 10.0}
        serializer = DeliverySerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['priority'], DEFAULT_DELIVERY_PRIORITY)
        self.assertFalse(serializer.validated_data['is_pickup'])
        self.assertEqual(serializer.validated_data['required_skills'], [])


class RouteOptimizationRequestSerializerTests(TestCase):
    def setUp(self):
        self.location_data = {"id": "loc1", "name": "L1", "latitude": 0.0, "longitude": 0.0}
        self.vehicle_data = {"id": "veh1", "capacity": 10.0, "start_location_id": "loc1"}
        self.delivery_data = {"id": "del1", "location_id": "loc1", "demand": 1.0}

    def test_valid_request(self):
        data = {
            "locations": [self.location_data],
            "vehicles": [self.vehicle_data],
            "deliveries": [self.delivery_data],
            "consider_traffic": True,
            "use_api": False,
            "api_key": "test_key"
        }
        serializer = RouteOptimizationRequestSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_request_missing_required_list_fields(self):
        data = {} # Missing locations, vehicles, deliveries
        serializer = RouteOptimizationRequestSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('locations', serializer.errors)
        self.assertIn('vehicles', serializer.errors)
        self.assertIn('deliveries', serializer.errors)

    def test_request_default_booleans(self):
        data = {
            "locations": [self.location_data],
            "vehicles": [self.vehicle_data],
            "deliveries": [self.delivery_data],
        }
        serializer = RouteOptimizationRequestSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertFalse(serializer.validated_data['consider_traffic'])
        self.assertFalse(serializer.validated_data['consider_time_windows'])
        self.assertTrue(serializer.validated_data['use_api']) # Default is True
        self.assertIsNone(serializer.validated_data.get('api_key'))
        self.assertIsNone(serializer.validated_data.get('traffic_data'))


class RouteSegmentSerializerTests(TestCase):
    def test_valid_segment(self):
        data = {
            "from_location": "A", "to_location": "B", "distance": 10.5, "estimated_time": 15.0,
            "path_coordinates": [[0.0,0.0],[1.0,1.0]], "traffic_factor": 1.2
        }
        serializer = RouteSegmentSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_segment_default_traffic_factor(self):
        data = {"from_location": "A", "to_location": "B", "distance": 10.5, "estimated_time": 15.0}
        serializer = RouteSegmentSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['traffic_factor'], 1.0)
        self.assertIsNone(serializer.validated_data.get('path_coordinates'))


class VehicleRouteSerializerTests(TestCase):
    def test_valid_vehicle_route(self):
        segment_data = {"from_location": "A", "to_location": "B", "distance": 10.0, "estimated_time": 20.0}
        data = {
            "vehicle_id": "veh1", "total_distance": 100.0, "total_time": 120.0,
            "stops": ["A", "B", "A"], "segments": [segment_data],
            "capacity_utilization": 0.75,
            "estimated_arrival_times": {"B": 60},
            "detailed_path": [[0.0,0.0],[1.0,1.0],[0.0,0.0]]
        }
        serializer = VehicleRouteSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_vehicle_route_optional_detailed_path(self):
        segment_data = {"from_location": "A", "to_location": "B", "distance": 10.0, "estimated_time": 20.0}
        data = {
            "vehicle_id": "veh1", "total_distance": 100.0, "total_time": 120.0,
            "stops": ["A", "B", "A"], "segments": [segment_data],
            "capacity_utilization": 0.75,
            "estimated_arrival_times": {"B": 60}
            # detailed_path is missing
        }
        serializer = VehicleRouteSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertIsNone(serializer.validated_data.get('detailed_path'))


class ReroutingInfoSerializerTests(TestCase):
    def test_valid_rerouting_info(self):
        data = {
            "reason": "traffic", "traffic_factors": 5, "completed_deliveries": 2,
            "remaining_deliveries": 3, "optimization_time_ms": 100
        }
        serializer = ReroutingInfoSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_rerouting_info_defaults(self):
        data = {"reason": "delay"}
        serializer = ReroutingInfoSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data['traffic_factors'], 0)
        self.assertEqual(serializer.validated_data['delay_locations'], 0)
        self.assertEqual(serializer.validated_data['blocked_segments'], 0)


class StatisticsSerializerTests(TestCase):
    def test_valid_statistics(self):
        rerouting_info_data = {"reason": "traffic"}
        data = {
            "used_vehicles": 2, "assigned_deliveries": 10,
            "computation_time_ms": 500,
            "rerouting_info": rerouting_info_data,
            "error": "None"
        }
        serializer = StatisticsSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertIsNotNone(serializer.validated_data['rerouting_info'])

    def test_statistics_all_optional(self):
        data = {}
        serializer = StatisticsSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors) # All fields are optional or have defaults if nested
        self.assertIsNone(serializer.validated_data.get('rerouting_info'))
        self.assertIsNone(serializer.validated_data.get('error'))


class OptimizationResultSerializerTests(TestCase):
    def setUp(self):
        self.valid_detailed_route_data = {
            "vehicle_id": "v1", "total_distance": 10, "total_time": 30,
            "stops": ["A", "B"], "segments": [], "capacity_utilization": 0.5,
            "estimated_arrival_times": {"B": 15}
        }
        self.valid_data_dict = {
            "status": "success",
            "routes": [], 
            "total_distance": 0.0,  
            "total_cost": 0.0,      
            "assigned_vehicles": {},
            "detailed_routes": [self.valid_detailed_route_data],
            "unassigned_deliveries": ["del3"],
            "statistics": None # Add default None as per serializer (allow_null=True)
        }

    @patch('route_optimizer.api.serializers.validate_optimization_result')
    def test_valid_optimization_result_dict(self, mock_validate):
        mock_validate.return_value = True # Assume core validation passes
        serializer = OptimizationResultSerializer(data=self.valid_data_dict)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        mock_validate.assert_called_once_with(self.valid_data_dict)

    @patch('route_optimizer.api.serializers.validate_optimization_result')
    def test_valid_optimization_result_dto_instance(self, mock_validate):
        mock_validate.return_value = True
        # Create an OptimizationResult DTO instance
        dto_instance = OptimizationResult(
            status="success",
            detailed_routes=[self.valid_detailed_route_data], # This should be a list of dicts
            unassigned_deliveries=["del3"]
        )
        # Initialize serializer with DTO instance for validation (atypical for .validate())
        serializer = OptimizationResultSerializer(data=dto_instance)

        # Manually call validate as is_valid() won't call it if instance is passed to __init__
        # and it's not what validate is typically for (it expects data dict)
        # Here, we test the explicit data=dto_instance scenario of the validate method
        validated_data = serializer.validate(dto_instance) # Test the branch for isinstance(data, OptimizationResult)

        self.assertIsNotNone(validated_data)
        mock_validate.assert_called_once_with(dataclasses.asdict(dto_instance))

    @patch('route_optimizer.api.serializers.validate_optimization_result')
    def test_invalid_optimization_result_core_validation_fails(self, mock_validate):
        mock_validate.side_effect = ValueError("Core validation failed")
        serializer = OptimizationResultSerializer(data=self.valid_data_dict)
        # The actual error message is now the generic one from the serializer's except block
        expected_message = "Invalid optimization result structure. Please ensure the data conforms to the required format."
        with self.assertRaisesMessage(serializers.ValidationError, expected_message):
            serializer.is_valid(raise_exception=True)
        mock_validate.assert_called_once_with(self.valid_data_dict)

    def test_optimization_result_invalid_data_type_for_validation(self):
        serializer = OptimizationResultSerializer() # No data initially
        with self.assertRaisesMessage(serializers.ValidationError, "Invalid data type for validation. Expected dict or OptimizationResult, got list."):
            serializer.validate([]) # Pass an invalid type to validate


class RouteOptimizationResponseSerializerTests(TestCase):
    def test_valid_response(self):
        vehicle_route_data = {
            "vehicle_id": "v1", "total_distance": 10, "total_time": 30,
            "stops": ["A", "B"], "segments": [], "capacity_utilization": 0.5,
            "estimated_arrival_times": {"B": 15}
        }
        data = {
            "status": "success", "total_distance": 100.0, "total_cost": 50.0,
            "routes": [vehicle_route_data], # This maps to detailed_routes from DTO
            "unassigned_deliveries": ["delX"],
            "statistics": {"used_vehicles": 1}
        }
        serializer = RouteOptimizationResponseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_response_routes_optional(self):
        data = {"status": "failed", "total_distance": 0.0, "total_cost": 0.0}
        serializer = RouteOptimizationResponseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertIsNone(serializer.validated_data.get('routes'))


class TrafficDataSerializerTests(TestCase):
    def test_valid_location_pairs(self):
        data = {"location_pairs": [["A", "B"]], "factors": [1.5]}
        serializer = TrafficDataSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_segments(self):
        data = {"segments": {"A-B": 1.5}}
        serializer = TrafficDataSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_empty(self): # Empty traffic data object
        data = {}
        serializer = TrafficDataSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_invalid_mismatched_pairs_factors(self):
        data = {"location_pairs": [["A", "B"]], "factors": [1.5, 2.0]}
        serializer = TrafficDataSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn("same number of elements", str(serializer.errors['non_field_errors']))

    def test_invalid_pairs_without_factors(self):
        data = {"location_pairs": [["A", "B"]]}
        serializer = TrafficDataSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn("If 'location_pairs' is provided, 'factors' must also be provided, and vice-versa.", str(serializer.errors['non_field_errors'][0]))


    def test_invalid_factors_without_pairs(self):
        data = {"factors": [1.5]}
        serializer = TrafficDataSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn("location_pairs' is provided", str(serializer.errors['non_field_errors']))

class ReroutingRequestSerializerTests(TestCase):
    def setUp(self):
        self.location_data = {"id": "loc1", "name": "L1", "latitude": 0.0, "longitude": 0.0}
        self.vehicle_data = {"id": "veh1", "capacity": 10.0, "start_location_id": "loc1"}
        self.delivery_data = {"id": "del1", "location_id": "loc1", "demand": 1.0}
        self.current_routes_json = { # Example of OptimizationResult as JSON
            "status": "success", "detailed_routes": [], "unassigned_deliveries": []
        }

    def test_valid_rerouting_traffic(self):
        data = {
            "current_routes": self.current_routes_json,
            "locations": [self.location_data],
            "vehicles": [self.vehicle_data],
            "original_deliveries": [self.delivery_data],
            "reroute_type": "traffic",
            "traffic_data": {"segments": {"loc1-loc2": 1.5}}
        }
        serializer = ReroutingRequestSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_rerouting_delay(self):
        data = {
            "current_routes": self.current_routes_json,
            "locations": [self.location_data],
            "vehicles": [self.vehicle_data],
            "original_deliveries": [self.delivery_data],
            "reroute_type": "delay",
            "delayed_location_ids": ["loc1"],
            "delay_minutes": {"loc1": 30}
        }
        serializer = ReroutingRequestSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_rerouting_roadblock(self):
        data = {
            "current_routes": self.current_routes_json,
            "locations": [self.location_data],
            "vehicles": [self.vehicle_data],
            "original_deliveries": [self.delivery_data],
            "reroute_type": "roadblock",
            "blocked_segments": [["loc1", "loc2"]]
        }
        serializer = ReroutingRequestSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_rerouting_missing_conditional_fields_validation(self):
        # Example: reroute_type is 'traffic' but no 'traffic_data'
        # The custom validate method allows this for now (passes if traffic_data is None/empty)
        data_traffic_no_data = {
            "current_routes": self.current_routes_json,
            "locations": [self.location_data], "vehicles": [self.vehicle_data], "original_deliveries": [self.delivery_data],
            "reroute_type": "traffic"
        }
        serializer = ReroutingRequestSerializer(data=data_traffic_no_data)
        self.assertTrue(serializer.is_valid(), serializer.errors) # Passes due to permissive validate

        # Example: reroute_type is 'delay' but no 'delayed_location_ids'
        data_delay_no_ids = {
            "current_routes": self.current_routes_json,
            "locations": [self.location_data], "vehicles": [self.vehicle_data], "original_deliveries": [self.delivery_data],
            "reroute_type": "delay", "delay_minutes": {"locX": 10}
        }
        serializer = ReroutingRequestSerializer(data=data_delay_no_ids)
        self.assertTrue(serializer.is_valid(), serializer.errors) # Also passes

    def test_rerouting_invalid_reroute_type(self):
        data = {
            "current_routes": self.current_routes_json,
            "locations": [self.location_data],
            "vehicles": [self.vehicle_data],
            "original_deliveries": [self.delivery_data],
            "reroute_type": "invalid_type"
        }
        serializer = ReroutingRequestSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('"invalid_type" is not a valid choice.', str(serializer.errors['reroute_type'][0]))
