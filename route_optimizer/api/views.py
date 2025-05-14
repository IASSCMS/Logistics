"""
API views for the route optimizer.

This module provides the API endpoints for the route optimization functionality.
"""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging
from typing import Dict, List, Tuple, Any, Optional # Ensure Tuple is imported

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.services.rerouting_service import ReroutingService
from route_optimizer.core.types_1 import Location, OptimizationResult # Import OptimizationResult DTO
from route_optimizer.models import Vehicle, Delivery
from route_optimizer.core.constants import DEFAULT_DELIVERY_PRIORITY # Import for default priority
from route_optimizer.api.serializers import (
    RouteOptimizationRequestSerializer,
    RouteOptimizationResponseSerializer,
    ReroutingRequestSerializer,
    # LocationSerializer, # Not directly used for DTO instantiation here
    # VehicleSerializer,  # Not directly used for DTO instantiation here
    # DeliverySerializer  # Not directly used for DTO instantiation here
)

# Set up logging
logger = logging.getLogger(__name__)

class OptimizeRoutesView(APIView):
    """
    API view for optimizing delivery routes.
    """
    
    @swagger_auto_schema(
        request_body=RouteOptimizationRequestSerializer,
        responses={200: RouteOptimizationResponseSerializer},
        operation_description="Optimize delivery routes based on provided locations, vehicles, and deliveries."
    )
    def post(self, request, format=None):
        """
        POST endpoint for route optimization.
        
        Args:
            request: HTTP request object containing route optimization parameters.
            format: Format of the response.
            
        Returns:
            Response object with optimization results.
        """
        serializer = RouteOptimizationRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Deserialize data into DTOs
            locations_data = serializer.validated_data['locations']
            locations = [
                Location(
                    id=loc_data['id'],
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    name=loc_data.get('name'),
                    address=loc_data.get('address'),
                    is_depot=loc_data.get('is_depot', False),
                    time_window_start=loc_data.get('time_window_start'),
                    time_window_end=loc_data.get('time_window_end'),
                    service_time=loc_data.get('service_time', 15)
                ) for loc_data in locations_data
            ]

            vehicles_data = serializer.validated_data['vehicles']
            vehicles = [
                Vehicle(
                    id=veh_data['id'],
                    capacity=veh_data['capacity'],
                    start_location_id=veh_data['start_location_id'],
                    end_location_id=veh_data.get('end_location_id'),
                    cost_per_km=veh_data.get('cost_per_km', 1.0),
                    fixed_cost=veh_data.get('fixed_cost', 0.0),
                    max_distance=veh_data.get('max_distance'),
                    max_stops=veh_data.get('max_stops'),
                    available=veh_data.get('available', True),
                    skills=veh_data.get('skills', [])
                ) for veh_data in vehicles_data
            ]

            deliveries_data = serializer.validated_data['deliveries']
            deliveries = [
                Delivery(
                    id=del_data['id'],
                    location_id=del_data['location_id'],
                    demand=del_data['demand'],
                    priority=del_data.get('priority', DEFAULT_DELIVERY_PRIORITY),
                    required_skills=del_data.get('required_skills', []),
                    is_pickup=del_data.get('is_pickup', False)
                ) for del_data in deliveries_data
            ]

            consider_traffic = serializer.validated_data.get('consider_traffic', False)
            consider_time_windows = serializer.validated_data.get('consider_time_windows', False)
            use_api = serializer.validated_data.get('use_api') # Let service handle default if None
            api_key = serializer.validated_data.get('api_key')
            
            traffic_data_input = serializer.validated_data.get('traffic_data')
            traffic_data_for_service: Optional[Dict[Tuple[int, int], float]] = None

            if consider_traffic and traffic_data_input:
                # Assuming traffic_data_input is JSON like:
                # {"location_pairs": [["id1","id2"], ...], "factors": [1.2, ...]}
                # or {"segments": {"id1-id2": 1.2, ...}}
                # This needs to be converted to Dict[Tuple[int, int], float] (index-based)
                # For a new optimization, this is less common unless pre-calculated
                # traffic factors for specific segments (by ID) are provided.
                # For now, let's assume it might come in the format RerouteView expects.
                
                # Example conversion if traffic_data_input is like TrafficDataSerializer
                # For an initial optimization, it might be simpler to pass segment IDs if not indices.
                # The current OptimizationService expects index-based traffic_data.
                # This conversion logic is complex without knowing the exact input format for initial optimization.
                # Placeholder: If you have a specific format, it would be converted here.
                # For simplicity, if traffic_data is passed, we assume it's already in the
                # Dict[Tuple[int,int], float] format or the service handles its conversion.
                # If it's JSON, it might look like what RerouteView handles:
                if isinstance(traffic_data_input, dict): # Check if it's a dict
                    temp_traffic_data: Dict[Tuple[int, int], float] = {}
                    location_id_to_idx = {loc.id: i for i, loc in enumerate(locations)}

                    # Scenario 1: From a list of pairs and factors
                    if 'location_pairs' in traffic_data_input and 'factors' in traffic_data_input:
                        pairs = traffic_data_input.get('location_pairs', [])
                        factors = traffic_data_input.get('factors', [])
                        for i, pair_ids in enumerate(pairs):
                            if i < len(factors) and len(pair_ids) == 2:
                                from_idx = location_id_to_idx.get(pair_ids[0])
                                to_idx = location_id_to_idx.get(pair_ids[1])
                                if from_idx is not None and to_idx is not None:
                                    temp_traffic_data[(from_idx, to_idx)] = factors[i]
                    # Scenario 2: From segments "from_id-to_id": factor
                    elif 'segments' in traffic_data_input and isinstance(traffic_data_input['segments'], dict):
                         for key, factor in traffic_data_input['segments'].items():
                            parts = key.split('-')
                            if len(parts) == 2:
                                from_idx = location_id_to_idx.get(parts[0])
                                to_idx = location_id_to_idx.get(parts[1])
                                if from_idx is not None and to_idx is not None:
                                    temp_traffic_data[(from_idx, to_idx)] = float(factor)
                    if temp_traffic_data:
                        traffic_data_for_service = temp_traffic_data
                    else:
                         # If traffic_data_input is directly Dict[Tuple[int,int], float] after JSON parsing (unlikely but possible)
                         # This part would need careful validation of keys and values.
                         # For now, we'll assume if it's not the list format, it's pre-processed or handled by service.
                         pass # Or log a warning if format is unexpected

            optimization_service = OptimizationService()
            result = optimization_service.optimize_routes(
                locations=locations,
                vehicles=vehicles,
                deliveries=deliveries,
                consider_traffic=consider_traffic,
                consider_time_windows=consider_time_windows,
                traffic_data=traffic_data_for_service,
                use_api=use_api,
                api_key=api_key
            )
            
            response_serializer = RouteOptimizationResponseSerializer(result)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("Error during new route optimization: %s", str(e))
            return Response(
                {"error": f"Optimization failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RerouteView(APIView):
    """
    API view for rerouting based on real-time events.
    """
    
    @swagger_auto_schema(
        request_body=ReroutingRequestSerializer,
        responses={200: RouteOptimizationResponseSerializer},
        operation_description="Reroute vehicles based on traffic, delays, or roadblocks."
    )
    def post(self, request, format=None):
        """
        POST endpoint for rerouting.
        
        Args:
            request: HTTP request object containing rerouting parameters.
            format: Format of the response.
            
        Returns:
            Response object with updated route plan.
        """
        serializer = ReroutingRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Deserialize basic data into DTOs
            locations_data = serializer.validated_data['locations']
            locations = [
                Location(
                    id=loc_data['id'],
                    name=loc_data.get('name'), # Use .get for optional fields
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    address=loc_data.get('address'),
                    is_depot=loc_data.get('is_depot', False),
                    time_window_start=loc_data.get('time_window_start'),
                    time_window_end=loc_data.get('time_window_end'),
                    service_time=loc_data.get('service_time', 15)
                ) for loc_data in locations_data
            ]
            
            vehicles_data = serializer.validated_data['vehicles']
            vehicles = [
                Vehicle(
                    id=veh_data['id'],
                    capacity=veh_data['capacity'],
                    start_location_id=veh_data['start_location_id'],
                    end_location_id=veh_data.get('end_location_id'),
                    cost_per_km=veh_data.get('cost_per_km', 1.0),
                    fixed_cost=veh_data.get('fixed_cost', 0.0),
                    max_distance=veh_data.get('max_distance'),
                    max_stops=veh_data.get('max_stops'),
                    available=veh_data.get('available', True),
                    skills=veh_data.get('skills', [])
                ) for veh_data in vehicles_data
            ]

            # CORRECTED: Convert original_deliveries from list of dicts to List[Delivery] DTOs
            original_deliveries_data = serializer.validated_data.get('original_deliveries', [])
            original_deliveries_dtos = [
                Delivery(
                    id=del_data['id'],
                    location_id=del_data['location_id'],
                    demand=del_data['demand'],
                    priority=del_data.get('priority', DEFAULT_DELIVERY_PRIORITY), 
                    required_skills=del_data.get('required_skills', []),
                    is_pickup=del_data.get('is_pickup', False)
                ) for del_data in original_deliveries_data
            ]
            
            # CORRECTED: Convert current_routes dict to OptimizationResult DTO
            current_routes_dict = serializer.validated_data['current_routes']
            current_routes_dto = OptimizationResult.from_dict(current_routes_dict)
            
            completed_deliveries = serializer.validated_data.get('completed_deliveries', [])
            reroute_type = serializer.validated_data.get('reroute_type', 'traffic')
            
            rerouting_service = ReroutingService()
            result: Optional[OptimizationResult] = None # Ensure result is defined
            
            if reroute_type == 'traffic':
                traffic_data_input = serializer.validated_data.get('traffic_data') # This comes as JSON parsed dict
                traffic_data_for_service: Dict[Tuple[int, int], float] = {}
                
                # Convert traffic data from JSON format to Dict[Tuple[int, int], float]
                # Assuming traffic_data_input is a dict like from TrafficDataSerializer:
                # e.g., {"location_pairs": [["id1","id2"], ["id2","id3"]], "factors": [1.5, 1.2]}
                # or {"segments": {"id1-id2": 1.5, "id2-id3": 1.2}}
                if traffic_data_input and isinstance(traffic_data_input, dict):
                    location_id_to_idx = {loc.id: i for i, loc in enumerate(locations)}
                    
                    if 'location_pairs' in traffic_data_input and 'factors' in traffic_data_input:
                        pairs = traffic_data_input.get('location_pairs', [])
                        factors = traffic_data_input.get('factors', [])
                        for i, pair_ids in enumerate(pairs):
                            if i < len(factors) and len(pair_ids) == 2:
                                from_idx = location_id_to_idx.get(pair_ids[0])
                                to_idx = location_id_to_idx.get(pair_ids[1])
                                if from_idx is not None and to_idx is not None:
                                    traffic_data_for_service[(from_idx, to_idx)] = float(factors[i])
                    elif 'segments' in traffic_data_input and isinstance(traffic_data_input['segments'], dict):
                         for key, factor in traffic_data_input['segments'].items():
                            parts = key.split('-') # Assuming "from_id-to_id" format
                            if len(parts) == 2:
                                from_idx = location_id_to_idx.get(parts[0])
                                to_idx = location_id_to_idx.get(parts[1])
                                if from_idx is not None and to_idx is not None:
                                    traffic_data_for_service[(from_idx, to_idx)] = float(factor)
                
                result = rerouting_service.reroute_for_traffic(
                    current_routes=current_routes_dto, # Pass DTO
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos, # Pass DTOs
                    completed_deliveries=completed_deliveries,
                    traffic_data=traffic_data_for_service
                )
            elif reroute_type == 'delay':
                delayed_location_ids = serializer.validated_data.get('delayed_location_ids', [])
                delay_minutes = serializer.validated_data.get('delay_minutes', {})
                result = rerouting_service.reroute_for_delay(
                    current_routes=current_routes_dto, # Pass DTO
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos, # Pass DTOs
                    completed_deliveries=completed_deliveries,
                    delayed_location_ids=delayed_location_ids,
                    delay_minutes=delay_minutes
                )
            elif reroute_type == 'roadblock':
                blocked_segments_input = serializer.validated_data.get('blocked_segments', [])
                blocked_segments = [tuple(segment) for segment in blocked_segments_input] # Ensure tuples
                result = rerouting_service.reroute_for_roadblock(
                    current_routes=current_routes_dto, # Pass DTO
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos, # Pass DTOs
                    completed_deliveries=completed_deliveries,
                    blocked_segments=blocked_segments
                )
            
            if result:
                response_serializer = RouteOptimizationResponseSerializer(result)
                return Response(response_serializer.data, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Invalid reroute type or no result obtained"}, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            logger.exception("Error during rerouting: %s", str(e))
            return Response(
                {"error": f"Rerouting failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
def health_check(request):
    """
    Health check endpoint to verify the API is running.
    
    Args:
        request: HTTP request object.
        
    Returns:
        Response object with health status.
    """
    return Response({"status": "healthy"}, status=status.HTTP_200_OK)