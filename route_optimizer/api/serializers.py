"""
Serializers for the route optimizer API.

This module provides serializers for converting between API requests/responses
and the internal data structures used by the route optimizer.
"""
from rest_framework import serializers
from typing import Dict, List, Any

from route_optimizer.core.types_1 import validate_optimization_result


class LocationSerializer(serializers.Serializer):
    """Serializer for Location objects."""
    id = serializers.CharField(max_length=100)
    name = serializers.CharField(max_length=255)
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    address = serializers.CharField(max_length=255, required=False, allow_null=True)
    is_depot = serializers.BooleanField(default=False)
    time_window_start = serializers.IntegerField(required=False, allow_null=True, 
                                              help_text="In minutes from midnight")
    time_window_end = serializers.IntegerField(required=False, allow_null=True,
                                            help_text="In minutes from midnight")
    service_time = serializers.IntegerField(default=15, help_text="Service time in minutes")


class VehicleSerializer(serializers.Serializer):
    """Serializer for Vehicle objects."""
    id = serializers.CharField(max_length=100)
    capacity = serializers.FloatField()
    start_location_id = serializers.CharField(max_length=100)
    end_location_id = serializers.CharField(max_length=100, required=False, allow_null=True)
    cost_per_km = serializers.FloatField(default=1.0)
    fixed_cost = serializers.FloatField(default=0.0)
    max_distance = serializers.FloatField(required=False, allow_null=True)
    max_stops = serializers.IntegerField(required=False, allow_null=True)
    available = serializers.BooleanField(default=True)
    skills = serializers.ListField(child=serializers.CharField(max_length=100), default=list)


class DeliverySerializer(serializers.Serializer):
    """Serializer for Delivery objects."""
    id = serializers.CharField(max_length=100)
    location_id = serializers.CharField(max_length=100)
    demand = serializers.FloatField()
    priority = serializers.IntegerField(default=1)
    required_skills = serializers.ListField(child=serializers.CharField(max_length=100), default=list)
    is_pickup = serializers.BooleanField(default=False)


class RouteOptimizationRequestSerializer(serializers.Serializer):
    """Serializer for route optimization requests."""
    locations = LocationSerializer(many=True)
    vehicles = VehicleSerializer(many=True)
    deliveries = DeliverySerializer(many=True)
    consider_traffic = serializers.BooleanField(default=False)
    consider_time_windows = serializers.BooleanField(default=False)
    use_api = serializers.BooleanField(default=True, required=False)
    api_key = serializers.CharField(max_length=255, required=False, allow_null=True)
    traffic_data = serializers.JSONField(required=False, allow_null=True)


class RouteSegmentSerializer(serializers.Serializer):
    """Serializer for a segment of a route."""
    from_location = serializers.CharField(max_length=100)
    to_location = serializers.CharField(max_length=100)
    distance = serializers.FloatField()
    estimated_time = serializers.FloatField(help_text="Estimated time in minutes")
    path_coordinates = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField(), min_length=2, max_length=2),
        required=False,
        help_text="List of [lat, lng] coordinates for detailed path"
    )
    traffic_factor = serializers.FloatField(default=1.0, help_text="Traffic multiplier for this segment")


class VehicleRouteSerializer(serializers.Serializer):
    """Serializer for a vehicle's route."""
    vehicle_id = serializers.CharField(max_length=100)
    total_distance = serializers.FloatField()
    total_time = serializers.FloatField(help_text="Total time in minutes")
    stops = serializers.ListField(child=serializers.CharField(max_length=100))
    segments = RouteSegmentSerializer(many=True)
    capacity_utilization = serializers.FloatField(help_text="Percentage of vehicle capacity used")
    estimated_arrival_times = serializers.DictField(
        child=serializers.IntegerField(),
        help_text="Mapping of location_id to arrival time in minutes from start"
    )
    detailed_path = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField(), min_length=2, max_length=2),
        required=False,
        help_text="Full detailed path as list of [lat, lng] coordinates"
    )


class ReroutingInfoSerializer(serializers.Serializer):
    """Serializer for rerouting information."""
    reason = serializers.CharField(max_length=50)
    traffic_factors = serializers.IntegerField(required=False, default=0)
    delayed_locations = serializers.IntegerField(required=False, default=0)
    blocked_segments = serializers.IntegerField(required=False, default=0)
    completed_deliveries = serializers.IntegerField(required=False, default=0)
    remaining_deliveries = serializers.IntegerField(required=False, default=0)
    optimization_time_ms = serializers.IntegerField(required=False)


class StatisticsSerializer(serializers.Serializer):
    """Serializer for optimization statistics."""
    total_vehicles = serializers.IntegerField(required=False)
    used_vehicles = serializers.IntegerField(required=False)
    total_deliveries = serializers.IntegerField(required=False)
    assigned_deliveries = serializers.IntegerField(required=False)
    total_distance = serializers.FloatField(required=False)
    total_time = serializers.FloatField(required=False)
    average_capacity_utilization = serializers.FloatField(required=False)
    computation_time_ms = serializers.IntegerField(required=False)
    rerouting_info = ReroutingInfoSerializer(required=False)
    error = serializers.CharField(required=False, allow_null=True)


class OptimizationResultSerializer(serializers.Serializer):
    """Serializer for OptimizationResult objects."""
    status = serializers.CharField(max_length=50)
    routes = serializers.ListField(child=serializers.ListField(
        child=serializers.CharField(max_length=100)
    ), required=False)
    total_distance = serializers.FloatField(required=False, default=0.0)
    total_cost = serializers.FloatField(required=False, default=0.0)
    assigned_vehicles = serializers.DictField(
        child=serializers.IntegerField(),
        required=False
    )
    unassigned_deliveries = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        default=list
    )
    detailed_routes = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list
    )
    statistics = StatisticsSerializer(required=False)

    def validate(self, data):
        """Custom validation for optimization result."""
        try:
            validate_optimization_result(data)
        except ValueError as e:
            raise serializers.ValidationError(str(e))
        return data


class RouteOptimizationResponseSerializer(OptimizationResultSerializer):
    """Serializer for route optimization responses."""
    status = serializers.CharField(max_length=50)
    total_distance = serializers.FloatField()
    total_cost = serializers.FloatField()
    routes = VehicleRouteSerializer(many=True, required=False)
    unassigned_deliveries = serializers.ListField(
        child=serializers.CharField(max_length=100), default=list
    )
    statistics = StatisticsSerializer(required=False)


class TrafficDataSerializer(serializers.Serializer):
    """Serializer for traffic data."""
    location_pairs = serializers.ListField(
        child=serializers.ListField(
            child=serializers.CharField(max_length=100),
            min_length=2,
            max_length=2
        ),
        required=False
    )
    factors = serializers.ListField(child=serializers.FloatField(), required=False)
    # Alternative structure
    segments = serializers.DictField(
        child=serializers.FloatField(),
        help_text="Dict mapping 'from_id-to_id' to traffic factor",
        required=False
    )


class ReroutingRequestSerializer(serializers.Serializer):
    """Serializer for rerouting requests."""
    current_routes = serializers.JSONField()
    locations = LocationSerializer(many=True)
    vehicles = VehicleSerializer(many=True)
    completed_deliveries = serializers.ListField(
        child=serializers.CharField(max_length=100), default=list
    )
    traffic_data = TrafficDataSerializer(required=False)
    delayed_location_ids = serializers.ListField(
        child=serializers.CharField(max_length=100), default=list
    )
    delay_minutes = serializers.DictField(
        child=serializers.IntegerField(),
        default=dict
    )
    blocked_segments = serializers.ListField(
        child=serializers.ListField(
            child=serializers.CharField(max_length=100),
            min_length=2,
            max_length=2
        ),
        default=list
    )
    reroute_type = serializers.ChoiceField(
        choices=['traffic', 'delay', 'roadblock'],
        default='traffic'
    )
    use_api = serializers.BooleanField(default=True, required=False)
    api_key = serializers.CharField(max_length=255, required=False, allow_null=True)