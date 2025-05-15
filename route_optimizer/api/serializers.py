"""
Serializers for the route optimizer API.

This module provides serializers for converting between API requests/responses
and the internal data structures used by the route optimizer.
"""
import dataclasses
import logging
from rest_framework import serializers
from typing import Dict, List, Any, Tuple 

from route_optimizer.core.types_1 import OptimizationResult, validate_optimization_result
from route_optimizer.core.constants import DEFAULT_DELIVERY_PRIORITY

logger = logging.getLogger(__name__)

class LocationSerializer(serializers.Serializer):
    """Serializer for Location objects."""
    id = serializers.CharField(max_length=100, help_text="Unique identifier for the location (e.g., 'depot', 'customer-123').")
    name = serializers.CharField(max_length=255, help_text="Human-readable name of the location.")
    latitude = serializers.FloatField(help_text="Latitude of the location in decimal degrees.")
    longitude = serializers.FloatField(help_text="Longitude of the location in decimal degrees.")
    address = serializers.CharField(max_length=255, required=False, allow_null=True, help_text="Full street address of the location (optional).")
    is_depot = serializers.BooleanField(default=False, help_text="True if this location is a depot, False otherwise.")
    time_window_start = serializers.IntegerField(required=False, allow_null=True, 
                                              help_text="Start of the time window for service at this location, in minutes from midnight (e.g., 540 for 9:00 AM).")
    time_window_end = serializers.IntegerField(required=False, allow_null=True,
                                            help_text="End of the time window for service at this location, in minutes from midnight (e.g., 1020 for 5:00 PM).")
    service_time = serializers.IntegerField(default=15, help_text="Time required for service at this location, in minutes (e.g., loading/unloading time). Default is 15 minutes.")


class VehicleSerializer(serializers.Serializer):
    """Serializer for Vehicle objects."""
    id = serializers.CharField(max_length=100, help_text="Unique identifier for the vehicle (e.g., 'vehicle-001').")
    capacity = serializers.FloatField(help_text="Capacity of the vehicle (e.g., weight, volume, number of items). Units must be consistent with delivery demands.")
    start_location_id = serializers.CharField(max_length=100, help_text="ID of the location where the vehicle starts its route.")
    end_location_id = serializers.CharField(max_length=100, required=False, allow_null=True, help_text="ID of the location where the vehicle must end its route. If null, defaults to start_location_id.")
    cost_per_km = serializers.FloatField(default=1.0, help_text="Cost incurred per kilometer traveled by this vehicle. Default is 1.0.")
    fixed_cost = serializers.FloatField(default=0.0, help_text="Fixed cost associated with using this vehicle for a route (e.g., daily rental cost). Default is 0.0.")
    max_distance = serializers.FloatField(required=False, allow_null=True, help_text="Maximum distance this vehicle can travel on a single route, in kilometers (optional).")
    max_stops = serializers.IntegerField(required=False, allow_null=True, help_text="Maximum number of stops this vehicle can make on a single route (optional).")
    available = serializers.BooleanField(default=True, help_text="True if the vehicle is available for use, False otherwise. Default is True.")
    skills = serializers.ListField(child=serializers.CharField(max_length=100), default=list, help_text="List of skills or capabilities this vehicle possesses (e.g., 'refrigeration', 'heavy_lift'). Default is an empty list.")

    class Meta:
        ref_name = 'RouteOptimizerVehicle' # Or any other unique name like 'RO_Vehicle'

class DeliverySerializer(serializers.Serializer):
    """Serializer for Delivery objects."""
    id = serializers.CharField(max_length=100, help_text="Unique identifier for the delivery or pickup task (e.g., 'order-456').")
    location_id = serializers.CharField(max_length=100, help_text="ID of the location where this delivery/pickup needs to occur.")
    demand = serializers.FloatField(help_text="Demand of this delivery. Positive for delivery (consumes capacity), could be negative for pickup (frees capacity, depending on solver configuration). Units must be consistent with vehicle capacity.")
    priority = serializers.IntegerField(default=DEFAULT_DELIVERY_PRIORITY, help_text="Priority of the delivery (e.g., 0=low, 1=normal, 2=high). Higher values typically indicate higher priority. Default is normal priority.")
    required_skills = serializers.ListField(child=serializers.CharField(max_length=100), default=list, help_text="List of skills required to perform this delivery (e.g., 'refrigeration'). Default is an empty list.")
    is_pickup = serializers.BooleanField(default=False, help_text="True if this task is a pickup, False if it's a delivery. Default is False.")


class RouteOptimizationRequestSerializer(serializers.Serializer):
    """Serializer for route optimization requests."""
    locations = LocationSerializer(many=True, help_text="List of all relevant location objects, including depots and customer sites.")
    vehicles = VehicleSerializer(many=True, help_text="List of all available vehicle objects.")
    deliveries = DeliverySerializer(many=True, help_text="List of all delivery or pickup tasks to be scheduled.")
    consider_traffic = serializers.BooleanField(default=False, help_text="If true, the optimizer will attempt to consider traffic conditions. Requires `traffic_data` or API usage. Default is False.")
    consider_time_windows = serializers.BooleanField(default=False, help_text="If true, the optimizer will respect the time windows specified for locations and potentially vehicles. Default is False.")
    use_api = serializers.BooleanField(default=True, required=False, help_text="If true, allows the optimizer to use external APIs (e.g., Google Maps) for distance/time calculations if configured. Default is True, but actual use depends on `api_key` and system settings.")
    api_key = serializers.CharField(max_length=255, required=False, allow_null=True, help_text="API key for external services (e.g., Google Maps API key), if overriding the system default or if one is not configured globally.")
    traffic_data = serializers.JSONField(required=False, allow_null=True, help_text="Optional. Pre-calculated traffic data. Format depends on service expectation, typically mapping segments (by ID or index) to factors. See `TrafficDataSerializer` for example structures.")


class RouteSegmentSerializer(serializers.Serializer):
    """Serializer for a segment of a route (i.e., travel between two consecutive stops)."""
    from_location = serializers.CharField(max_length=100, help_text="ID of the origin location for this segment.")
    to_location = serializers.CharField(max_length=100, help_text="ID of the destination location for this segment.")
    distance = serializers.FloatField(help_text="Distance of this segment in kilometers.")
    estimated_time = serializers.FloatField(help_text="Estimated travel time for this segment in minutes.")
    path_coordinates = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField(), min_length=2, max_length=2),
        required=False, allow_null=True, # allow_null for cases where it's not generated
        help_text="List of [latitude, longitude] coordinates representing the detailed path for this segment (optional)."
    )
    traffic_factor = serializers.FloatField(default=1.0, help_text="Traffic multiplier applied to this segment. 1.0 means no traffic impact. Default is 1.0.")


class VehicleRouteSerializer(serializers.Serializer):
    """Serializer for a single vehicle's complete optimized route."""
    vehicle_id = serializers.CharField(max_length=100, help_text="ID of the vehicle assigned to this route.")
    total_distance = serializers.FloatField(help_text="Total distance of this vehicle's route in kilometers.")
    total_time = serializers.FloatField(help_text="Total estimated time for this vehicle's route in minutes (including travel and service times).")
    stops = serializers.ListField(child=serializers.CharField(max_length=100), help_text="Ordered list of location IDs visited by this vehicle, including start and end depots.")
    segments = RouteSegmentSerializer(many=True, help_text="List of route segments that make up this vehicle's path.")
    capacity_utilization = serializers.FloatField(help_text="Percentage of the vehicle's capacity utilized on this route (e.g., 0.75 for 75% used).")
    estimated_arrival_times = serializers.DictField(
        child=serializers.IntegerField(), # Assuming arrival times are in minutes from a common epoch (e.g., route start or midnight)
        help_text="Mapping of location_id to its estimated arrival time in minutes (e.g., from route start or midnight, ensure consistency)."
    )
    detailed_path = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField(), min_length=2, max_length=2),
        required=False, allow_null=True,
        help_text="Full detailed path for the entire vehicle route as a list of [latitude, longitude] coordinates (optional, can be large)."
    )


class ReroutingInfoSerializer(serializers.Serializer):
    """Serializer for information specific to a rerouting operation."""
    reason = serializers.CharField(max_length=50, help_text="Reason for the rerouting (e.g., 'traffic', 'service_delay', 'roadblock').")
    traffic_factors = serializers.IntegerField(required=False, default=0, help_text="Count of distinct traffic factors or segments considered during traffic rerouting.")
    delay_locations = serializers.IntegerField(required=False, default=0, help_text="Count of locations that reported delays leading to delay rerouting.") # Consider changing to List[str] if actual IDs are more useful
    blocked_segments = serializers.IntegerField(required=False, default=0, help_text="Count of road segments that were blocked, leading to roadblock rerouting.") # Consider changing to List[List[str]]
    completed_deliveries = serializers.IntegerField(required=False, default=0, help_text="Number of deliveries completed before this rerouting was initiated.")
    remaining_deliveries = serializers.IntegerField(required=False, default=0, help_text="Number of deliveries remaining after this rerouting was initiated.")
    optimization_time_ms = serializers.IntegerField(required=False, help_text="Time taken for the rerouting optimization process in milliseconds (optional).")


class StatisticsSerializer(serializers.Serializer):
    """Serializer for overall optimization statistics."""
    total_vehicles = serializers.IntegerField(required=False, help_text="Total number of vehicles available for the optimization problem.")
    used_vehicles = serializers.IntegerField(required=False, help_text="Number of vehicles actually used in the optimized solution.")
    total_deliveries = serializers.IntegerField(required=False, help_text="Total number of deliveries in the optimization problem.")
    assigned_deliveries = serializers.IntegerField(required=False, help_text="Number of deliveries successfully assigned to routes.")
    total_distance = serializers.FloatField(required=False, help_text="Total distance covered by all routes in the solution, in kilometers.")
    total_time = serializers.FloatField(required=False, help_text="Total time for all routes in the solution, in minutes (sum of vehicle route times).")
    average_capacity_utilization = serializers.FloatField(required=False, help_text="Average capacity utilization across all used vehicles (e.g., 0.6 for 60%).")
    computation_time_ms = serializers.IntegerField(required=False, help_text="Time taken for the core optimization computation in milliseconds.")
    rerouting_info = ReroutingInfoSerializer(required=False, allow_null=True, help_text="Specific information if this result is from a rerouting operation (optional).")
    error = serializers.CharField(required=False, allow_null=True, help_text="Error message if the optimization failed or encountered issues.")


class OptimizationResultSerializer(serializers.Serializer):
    """Base serializer for OptimizationResult DTO, often used for internal representation or as a base for responses."""
    status = serializers.CharField(max_length=50, help_text="Status of the optimization ('success', 'failed', 'error').")
    routes = serializers.ListField(
        child=serializers.ListField(child=serializers.CharField(max_length=100)), 
        required=False, 
        help_text="Simplified list of routes, where each route is a list of location IDs. More detailed routes are in 'detailed_routes'."
    )
    total_distance = serializers.FloatField(required=False, default=0.0, help_text="Overall total distance of all routes in kilometers (may be refined in statistics).")
    total_cost = serializers.FloatField(required=False, default=0.0, help_text="Overall total cost of all routes (may be refined in statistics).")
    assigned_vehicles = serializers.DictField(
        child=serializers.IntegerField(),
        required=False,
        help_text="Mapping of vehicle IDs to the index of the route they are assigned to in the 'routes' or 'detailed_routes' list."
    )
    unassigned_deliveries = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        default=list,
        help_text="List of delivery IDs that could not be assigned to any route."
    )
    detailed_routes = serializers.ListField( # This now matches VehicleRouteSerializer structure more closely
        child=VehicleRouteSerializer(),     # Use VehicleRouteSerializer for each item
        required=False,
        default=list,
        help_text="List of detailed routes, each providing comprehensive information for a single vehicle's journey."
    )
    statistics = StatisticsSerializer(required=False, allow_null=True, help_text="Additional statistics about the optimization result.")

    def validate(self, data: Any) -> Any:
        """
        Custom validation for optimization result structure.
        This method primarily expects 'data' to be a dictionary for input validation.
        It includes a safeguard to handle OptimizationResult DTO instances if is_valid()
        is called on a serializer initialized with an instance (which is uncommon for 'validate').
        """
        data_to_validate: Dict[str, Any]
        if isinstance(data, OptimizationResult):
            # This case is less common for a .validate() call but handled for robustness.
            logger.warning(
                "OptimizationResultSerializer.validate received an OptimizationResult DTO instance. "
                "Converting to dict for validation. This is an atypical use of .validate()."
            )
            # Recursively convert DTO to dict for validation
            data_to_validate = dataclasses.asdict(data)
        elif isinstance(data, dict):
            data_to_validate = data
        else:
            # If data is neither a dict nor an OptimizationResult DTO, it's an invalid type for this validation.
            raise serializers.ValidationError(
                f"Invalid data type for validation. Expected dict or OptimizationResult, got {type(data).__name__}."
            )
        
        try:
            # validate_optimization_result expects a dictionary
            validate_optimization_result(data_to_validate)
        except ValueError as e:
            # Log the detailed error for internal review
            logger.error(f"Validation error in OptimizationResult data: {str(e)}", exc_info=True)
            # Raise a generic validation error to the client
            raise serializers.ValidationError("Invalid optimization result structure. Please ensure the data conforms to the required format.")     
        
        # The .validate() method must return the validated data (the original 'data' argument)
        return data

# RouteOptimizationResponseSerializer needs to align with OptimizationResult DTO structure
# and how `VehicleRouteSerializer` structures individual routes.
class RouteOptimizationResponseSerializer(serializers.Serializer): # Changed from inheriting OptimizationResultSerializer for clarity
    """Serializer for the final route optimization response, aligning with OptimizationResult DTO."""
    status = serializers.CharField(max_length=50, help_text="Status of the optimization ('success', 'failed', 'error').")
    total_distance = serializers.FloatField(help_text="Overall total distance of all optimized routes in kilometers.")
    total_cost = serializers.FloatField(help_text="Overall total cost of all optimized routes.")
    routes = VehicleRouteSerializer(many=True, required=False, help_text="List of detailed vehicle routes. This is the primary output for successful optimizations. Renamed from 'detailed_routes' in OptimizationResult DTO for client clarity, but maps to it.") # Maps to OptimizationResult.detailed_routes
    unassigned_deliveries = serializers.ListField(
        child=serializers.CharField(max_length=100), 
        default=list, 
        help_text="List of delivery IDs that could not be assigned to any route."
    )
    statistics = StatisticsSerializer(required=False, allow_null=True, help_text="Additional statistics about the optimization result.")
    
    # If you need to map from OptimizationResult DTO to this serializer's field names,
    # you might override to_representation or ensure field names match the DTO attributes.
    # For example, if OptimizationResult DTO has 'detailed_routes' but response has 'routes':
    # In to_representation: representation['routes'] = representation.pop('detailed_routes', [])
    # Or, even better, make the DTO's detailed_routes structure match VehicleRouteSerializer directly,
    # and if the response field is 'routes', then `RouteOptimizationResponseSerializer(source='detailed_routes', many=True)`
    # For now, I've made `OptimizationResultSerializer.detailed_routes` use `VehicleRouteSerializer`.
    # And `RouteOptimizationResponseSerializer.routes` directly use `VehicleRouteSerializer` for clarity.
    # This means the OptimizationResult DTO's `detailed_routes` field should contain list of dicts that `VehicleRouteSerializer` can handle.
    # The `OptimizationService` populates `detailed_routes` with dicts which are compatible.


class TrafficDataSerializer(serializers.Serializer):
    """
    Serializer for specifying traffic data input. 
    Traffic can be specified by pairs of location IDs and corresponding factors,
    or by segments identified by a 'from_id-to_id' key.
    """
    location_pairs = serializers.ListField(
        child=serializers.ListField(
            child=serializers.CharField(max_length=100, help_text="Location ID."),
            min_length=2,
            max_length=2,
            help_text="A pair of [from_location_id, to_location_id]."
        ),
        required=False,
        help_text="List of location ID pairs. The order should match the 'factors' list."
    )
    factors = serializers.ListField(
        child=serializers.FloatField(help_text="Traffic factor (e.g., 1.0 = no impact, 1.5 = 50% slower)."), 
        required=False,
        help_text="List of traffic factors corresponding to 'location_pairs'. Must be same length as 'location_pairs'."
    )
    segments = serializers.DictField(
        child=serializers.FloatField(help_text="Traffic factor."),
        help_text="Alternative. Dictionary mapping segment keys (e.g., 'from_loc_id-to_loc_id') to traffic factors.",
        required=False
    )

    def validate(self, data):
        if 'location_pairs' in data and 'factors' in data:
            if len(data['location_pairs']) != len(data['factors']):
                raise serializers.ValidationError("If 'location_pairs' and 'factors' are provided, they must have the same number of elements.")
        elif ('location_pairs' in data and 'factors' not in data) or \
             ('location_pairs' not in data and 'factors' in data):
            raise serializers.ValidationError("If 'location_pairs' is provided, 'factors' must also be provided, and vice-versa.")
        if not data.get('location_pairs') and not data.get('segments'):
             # Allow empty traffic data if neither is provided, but if traffic_data itself is provided, one form should exist.
             # This depends on whether an empty traffic_data object is valid or should imply no traffic_data was sent.
             # If traffic_data is optional at request level, this might be fine.
             pass
        return data


class ReroutingRequestSerializer(serializers.Serializer):
    """Serializer for rerouting requests."""
    current_routes = serializers.JSONField(help_text="The current route plan (OptimizationResult) as a JSON object, which needs to be adjusted.")
    locations = LocationSerializer(many=True, help_text="Full list of relevant location DTOs for the rerouting context.")
    vehicles = VehicleSerializer(many=True, help_text="Full list of relevant vehicle DTOs for the rerouting context.")
    original_deliveries = DeliverySerializer(many=True, help_text="The full list of original delivery objects relevant to the current_routes. This is used to determine remaining deliveries and map delivery IDs to locations.")
    
    completed_deliveries = serializers.ListField(
        child=serializers.CharField(max_length=100), 
        required=False, 
        default=list,
        help_text="List of delivery IDs that have been completed since the 'current_routes' plan was generated."
    )
    reroute_type = serializers.ChoiceField(
        choices=['traffic', 'delay', 'roadblock'], 
        default='traffic',
        help_text="The type of event triggering the reroute: 'traffic', 'service_delay', or 'roadblock'."
    )
    
    # Fields for traffic rerouting
    traffic_data = TrafficDataSerializer(required=False, allow_null=True, help_text="Traffic data relevant for 'traffic' reroute_type. See TrafficDataSerializer for format.") # Use the dedicated serializer
    
    # Fields for delay rerouting
    delayed_location_ids = serializers.ListField(
        child=serializers.CharField(max_length=100), 
        required=False, 
        default=list,
        help_text="List of location IDs experiencing service delays (for 'delay' reroute_type)."
    )
    delay_minutes = serializers.DictField(
        child=serializers.IntegerField(min_value=0), 
        required=False,
        default=dict,
        help_text="Dictionary mapping delayed_location_ids to the additional delay in minutes (for 'delay' reroute_type)."
    )
    # Fields for roadblock rerouting
    blocked_segments = serializers.ListField(
        child=serializers.ListField(
            child=serializers.CharField(max_length=100, help_text="Location ID."),
            min_length=2,
            max_length=2,
            help_text="A pair representing a blocked segment: [from_location_id, to_location_id]."
        ),
        required=False, 
        default=list,
        help_text="List of blocked road segments, where each segment is a [from_location_id, to_location_id] pair (for 'roadblock' reroute_type)."
    )
    # These were duplicated, `use_api` and `api_key` are more relevant to initial optimization
    # but could be passed for rerouting if the rerouting itself might involve API calls for matrix regeneration.
    # For now, let ReroutingService decide if it needs them internally from OptimizationService.
    # use_api = serializers.BooleanField(default=True, required=False)
    # api_key = serializers.CharField(max_length=255, required=False, allow_null=True)

    def validate(self, data):
        reroute_type = data.get('reroute_type')
        if reroute_type == 'traffic' and not data.get('traffic_data'):
            # Depending on strictness, might require traffic_data if type is traffic
            # For now, assume it can be an empty object if no specific factors are provided
            pass
        if reroute_type == 'delay' and (not data.get('delayed_location_ids') or not data.get('delay_minutes')):
            # If type is delay, expect delay_location_ids and delay_minutes.
            # Allowing empty lists/dicts might be okay if no specific delays are known yet but want to trigger time-window re-eval.
            pass
        if reroute_type == 'roadblock' and not data.get('blocked_segments'):
            # Similar to above, allow empty if no specific blocks.
            pass
        return data