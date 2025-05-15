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
        request_body=RouteOptimizationRequestSerializer, # Defines the expected input
        responses={
            200: RouteOptimizationResponseSerializer, # Defines the successful output
            400: "Bad Request - Invalid input data", # Example for error response
            500: "Internal Server Error - Optimization failed" # Example for error response
        },
        operation_id="optimize_routes_post", # Optional: A unique ID for the operation
        operation_description="""Initiates a new route optimization plan based on provided locations, vehicles, and deliveries. 
        Considers constraints like vehicle capacities, time windows (if specified), and traffic conditions (if specified).""",
        tags=['Route Optimization']
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
            logger.error(f"OptimizeRoutesView validation error: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Deserialize data into DTOs
            locations_data = serializer.validated_data['locations']
            locations = [
                Location(**loc_data) for loc_data in locations_data
            ]

            vehicles_data = serializer.validated_data['vehicles']
            vehicles = [
                Vehicle(**veh_data) for veh_data in vehicles_data
            ]

            deliveries_data = serializer.validated_data['deliveries']
            deliveries = [
                Delivery(**del_data) for del_data in deliveries_data
            ]

            consider_traffic = serializer.validated_data.get('consider_traffic', False)
            consider_time_windows = serializer.validated_data.get('consider_time_windows', False)
            use_api = serializer.validated_data.get('use_api') # Let service handle default if None
            api_key = serializer.validated_data.get('api_key')
            
            traffic_data_input = serializer.validated_data.get('traffic_data')
            traffic_data_for_service: Optional[Dict[Tuple[int, int], float]] = None

            if consider_traffic and traffic_data_input:
                # Convert JSON traffic_data to service-expected Dict[Tuple[int, int], float]
                # This assumes traffic_data_input is structured as per TrafficDataSerializer
                # (e.g., {"location_pairs": [["id1","id2"], ...], "factors": [1.2, ...]} or {"segments": {"id1-id2": 1.2}})
                # The OptimizationService expects index-based keys.
                temp_traffic_data: Dict[Tuple[int, int], float] = {}
                location_id_to_idx = {loc.id: i for i, loc in enumerate(locations)}

                if 'location_pairs' in traffic_data_input and 'factors' in traffic_data_input:
                    pairs = traffic_data_input.get('location_pairs', [])
                    factors = traffic_data_input.get('factors', [])
                    for i, pair_ids in enumerate(pairs):
                        if i < len(factors) and len(pair_ids) == 2:
                            from_idx = location_id_to_idx.get(pair_ids[0])
                            to_idx = location_id_to_idx.get(pair_ids[1])
                            if from_idx is not None and to_idx is not None:
                                temp_traffic_data[(from_idx, to_idx)] = float(factors[i])
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
                    logger.warning("Traffic data provided but not in a recognized format for initial optimization.")

            optimization_service = OptimizationService()
            result_dto = optimization_service.optimize_routes(
                locations=locations,
                vehicles=vehicles,
                deliveries=deliveries,
                consider_traffic=consider_traffic,
                consider_time_windows=consider_time_windows,
                traffic_data=traffic_data_for_service,
                use_api=use_api,
                api_key=api_key
            )
            
            # Map OptimizationResult DTO to RouteOptimizationResponseSerializer
            # The serializer now expects `routes` to be its main detailed route list.
            # OptimizationResult DTO stores this in `detailed_routes`.
            response_data = {
                "status": result_dto.status,
                "total_distance": result_dto.total_distance,
                "total_cost": result_dto.total_cost,
                "routes": result_dto.detailed_routes, # Map DTO's detailed_routes to serializer's routes
                "unassigned_deliveries": result_dto.unassigned_deliveries,
                "statistics": result_dto.statistics
            }
            response_serializer = RouteOptimizationResponseSerializer(data=response_data)
            if not response_serializer.is_valid(): # Should be valid if DTO is correct
                logger.error(f"OptimizeRoutesView response serialization error: {response_serializer.errors}")
                return Response(response_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as e: # This catches unexpected server errors
            logger.exception("Critical error during new route optimization: %s", str(e)) # Logger already captures the full str(e) and stack trace
            return Response(
                {"error": "An unexpected error occurred during route optimization. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RerouteView(APIView):
    """
    API view for rerouting based on real-time events.
    """
    
    @swagger_auto_schema(
        request_body=ReroutingRequestSerializer,
        responses={
            200: openapi.Response("Successful rerouting.", RouteOptimizationResponseSerializer),
            400: openapi.Response("Bad Request - Invalid input data. Check serializer errors."),
            500: openapi.Response("Internal Server Error - Rerouting process failed.")
        },
        operation_id="reroute_vehicles_update",
        operation_description="""Dynamically reroutes vehicles based on real-time events such as traffic updates, service delays, or roadblocks. 
        Requires the current route plan and event-specific data.""",
        tags=['Route Rerouting']
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
            logger.error(f"RerouteView validation error: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            locations_data = serializer.validated_data['locations']
            locations = [Location(**loc_data) for loc_data in locations_data]
            
            vehicles_data = serializer.validated_data['vehicles']
            vehicles = [Vehicle(**veh_data) for veh_data in vehicles_data]

            original_deliveries_data = serializer.validated_data.get('original_deliveries', [])
            original_deliveries_dtos = [Delivery(**del_data) for del_data in original_deliveries_data]
            
            current_routes_dict = serializer.validated_data['current_routes']
            current_routes_dto = OptimizationResult.from_dict(current_routes_dict) # Use static method
            
            completed_deliveries = serializer.validated_data.get('completed_deliveries', [])
            reroute_type = serializer.validated_data.get('reroute_type', 'traffic')
            
            rerouting_service = ReroutingService()
            result_dto: Optional[OptimizationResult] = None 
            
            if reroute_type == 'traffic':
                traffic_data_input = serializer.validated_data.get('traffic_data', {}) # Default to empty dict
                traffic_data_for_service: Dict[Tuple[int, int], float] = {}
                
                if traffic_data_input: # traffic_data_input is already a dict from TrafficDataSerializer
                    location_id_to_idx = {loc.id: i for i, loc in enumerate(locations)}
                    if 'location_pairs' in traffic_data_input and 'factors' in traffic_data_input:
                        # ... (same conversion logic as in OptimizeRoutesView) ...
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
                            parts = key.split('-')
                            if len(parts) == 2:
                                from_idx = location_id_to_idx.get(parts[0])
                                to_idx = location_id_to_idx.get(parts[1])
                                if from_idx is not None and to_idx is not None:
                                    traffic_data_for_service[(from_idx, to_idx)] = float(factor)
                
                result_dto = rerouting_service.reroute_for_traffic(
                    current_routes=current_routes_dto,
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos,
                    completed_deliveries=completed_deliveries,
                    traffic_data=traffic_data_for_service
                )
            elif reroute_type == 'delay':
                delayed_location_ids = serializer.validated_data.get('delayed_location_ids', [])
                delay_minutes = serializer.validated_data.get('delay_minutes', {})
                result_dto = rerouting_service.reroute_for_delay(
                    current_routes=current_routes_dto, 
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos,
                    completed_deliveries=completed_deliveries,
                    delayed_location_ids=delayed_location_ids,
                    delay_minutes=delay_minutes
                )
            elif reroute_type == 'roadblock':
                blocked_segments_input = serializer.validated_data.get('blocked_segments', [])
                # blocked_segments in ReroutingRequestSerializer is List[List[str]]
                # ReroutingService.reroute_for_roadblock expects List[Tuple[str, str]]
                blocked_segments_tuples = [tuple(segment) for segment in blocked_segments_input]
                result_dto = rerouting_service.reroute_for_roadblock(
                    current_routes=current_routes_dto,
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos,
                    completed_deliveries=completed_deliveries,
                    blocked_segments=blocked_segments_tuples
                )
            
            if result_dto:
                response_data = {
                    "status": result_dto.status,
                    "total_distance": result_dto.total_distance,
                    "total_cost": result_dto.total_cost,
                    "routes": result_dto.detailed_routes, # Map DTO's detailed_routes to serializer's routes
                    "unassigned_deliveries": result_dto.unassigned_deliveries,
                    "statistics": result_dto.statistics
                }
                response_serializer = RouteOptimizationResponseSerializer(data=response_data)
                if not response_serializer.is_valid():
                     logger.error(f"RerouteView response serialization error: {response_serializer.errors}")
                     return Response(response_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                return Response(response_serializer.data, status=status.HTTP_200_OK)
            else:
                # This case should ideally be handled by exceptions in ReroutingService returning an error DTO
                logger.error("Rerouting did not produce a result DTO for an unknown reason.")
                return Response({"error": "Invalid reroute type or no result obtained from rerouting service."}, status=status.HTTP_400_BAD_REQUEST)
            
        # Suggested change:
        except Exception as e:
            logger.exception("Critical error during rerouting: %s", str(e)) # Logger already captures the full str(e) and stack trace
            return Response(
                {"error": "An unexpected error occurred during rerouting. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

"""
API views for the route optimizer.
"""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi # Make sure openapi is imported
import logging
from typing import Dict, List, Tuple, Any, Optional

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.services.rerouting_service import ReroutingService
from route_optimizer.core.types_1 import Location, OptimizationResult # Import DTOs
from route_optimizer.models import Vehicle, Delivery # Import dataclasses
from route_optimizer.core.constants import DEFAULT_DELIVERY_PRIORITY
from route_optimizer.api.serializers import (
    RouteOptimizationRequestSerializer,
    RouteOptimizationResponseSerializer,
    ReroutingRequestSerializer
)

logger = logging.getLogger(__name__)

class OptimizeRoutesView(APIView):
    """API view for optimizing new delivery routes."""
    
    @swagger_auto_schema(
        request_body=RouteOptimizationRequestSerializer,
        responses={
            200: openapi.Response("Successful optimization.", RouteOptimizationResponseSerializer),
            400: openapi.Response("Bad Request - Invalid input data. Check serializer errors."),
            500: openapi.Response("Internal Server Error - Optimization process failed.")
        },
        operation_id="optimize_routes_create",
        operation_description="""Initiates a new route optimization plan based on provided locations, vehicles, and deliveries. 
        Considers constraints like vehicle capacities, time windows (if specified), and traffic conditions (if specified).""",
        tags=['Route Optimization']
    )
    def post(self, request, format=None):
        serializer = RouteOptimizationRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"OptimizeRoutesView validation error: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            locations_data = serializer.validated_data['locations']
            locations = [
                Location(**loc_data) for loc_data in locations_data
            ]

            vehicles_data = serializer.validated_data['vehicles']
            vehicles = [
                Vehicle(**veh_data) for veh_data in vehicles_data
            ]

            deliveries_data = serializer.validated_data['deliveries']
            deliveries = [
                Delivery(**del_data) for del_data in deliveries_data
            ]

            consider_traffic = serializer.validated_data.get('consider_traffic', False)
            consider_time_windows = serializer.validated_data.get('consider_time_windows', False)
            use_api = serializer.validated_data.get('use_api')
            api_key = serializer.validated_data.get('api_key')
            
            traffic_data_input = serializer.validated_data.get('traffic_data')
            traffic_data_for_service: Optional[Dict[Tuple[int, int], float]] = None

            if consider_traffic and traffic_data_input:
                # Convert JSON traffic_data to service-expected Dict[Tuple[int, int], float]
                # This assumes traffic_data_input is structured as per TrafficDataSerializer
                # (e.g., {"location_pairs": [["id1","id2"], ...], "factors": [1.2, ...]} or {"segments": {"id1-id2": 1.2}})
                # The OptimizationService expects index-based keys.
                temp_traffic_data: Dict[Tuple[int, int], float] = {}
                location_id_to_idx = {loc.id: i for i, loc in enumerate(locations)}

                if 'location_pairs' in traffic_data_input and 'factors' in traffic_data_input:
                    pairs = traffic_data_input.get('location_pairs', [])
                    factors = traffic_data_input.get('factors', [])
                    for i, pair_ids in enumerate(pairs):
                        if i < len(factors) and len(pair_ids) == 2:
                            from_idx = location_id_to_idx.get(pair_ids[0])
                            to_idx = location_id_to_idx.get(pair_ids[1])
                            if from_idx is not None and to_idx is not None:
                                temp_traffic_data[(from_idx, to_idx)] = float(factors[i])
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
                    logger.warning("Traffic data provided but not in a recognized format for initial optimization.")


            optimization_service = OptimizationService()
            result_dto = optimization_service.optimize_routes(
                locations=locations,
                vehicles=vehicles,
                deliveries=deliveries,
                consider_traffic=consider_traffic,
                consider_time_windows=consider_time_windows,
                traffic_data=traffic_data_for_service,
                use_api=use_api,
                api_key=api_key
            )
            
            # Map OptimizationResult DTO to RouteOptimizationResponseSerializer
            # The serializer now expects `routes` to be its main detailed route list.
            # OptimizationResult DTO stores this in `detailed_routes`.
            response_data = {
                "status": result_dto.status,
                "total_distance": result_dto.total_distance,
                "total_cost": result_dto.total_cost,
                "routes": result_dto.detailed_routes,
                "unassigned_deliveries": result_dto.unassigned_deliveries,
                "statistics": result_dto.statistics
            }
            response_serializer = RouteOptimizationResponseSerializer(data=response_data)
            
            if not response_serializer.is_valid():
                logger.error(f"OptimizeRoutesView response serialization error: {response_serializer.errors}")
                return Response(response_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            http_status_to_return = status.HTTP_200_OK
            if result_dto.status != 'success':
                # If the service indicates an error or failure (e.g., invalid inputs like no locations,
                # or no solution found), it's often a client-side correctable issue.
                http_status_to_return = status.HTTP_400_BAD_REQUEST
                # You might want more granular control, e.g., specific errors from service mapping to 500.
                # For now, non-success from service DTO implies a 400.
            
            return Response(response_serializer.data, status=http_status_to_return)

        except Exception as e: # This catches unexpected server errors
            logger.exception("Critical error during new route optimization: %s", str(e)) # Logger already captures the full str(e) and stack trace
            return Response(
                {"error": "An unexpected error occurred during route optimization. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RerouteView(APIView):
    """API view for rerouting vehicles based on real-time events."""
    
    @swagger_auto_schema(
        request_body=ReroutingRequestSerializer,
        responses={
            200: openapi.Response("Successful rerouting.", RouteOptimizationResponseSerializer),
            400: openapi.Response("Bad Request - Invalid input data. Check serializer errors."),
            500: openapi.Response("Internal Server Error - Rerouting process failed.")
        },
        operation_id="reroute_vehicles_update",
        operation_description="""Dynamically reroutes vehicles based on real-time events such as traffic updates, service delays, or roadblocks. 
        Requires the current route plan and event-specific data.""",
        tags=['Route Rerouting']
    )
    def post(self, request, format=None):
        serializer = ReroutingRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"RerouteView validation error: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            locations_data = serializer.validated_data['locations']
            locations = [Location(**loc_data) for loc_data in locations_data]
            
            vehicles_data = serializer.validated_data['vehicles']
            vehicles = [Vehicle(**veh_data) for veh_data in vehicles_data]

            original_deliveries_data = serializer.validated_data.get('original_deliveries', [])
            original_deliveries_dtos = [Delivery(**del_data) for del_data in original_deliveries_data]
            
            current_routes_dict = serializer.validated_data['current_routes']
            current_routes_dto = OptimizationResult.from_dict(current_routes_dict) # Use static method
            
            completed_deliveries = serializer.validated_data.get('completed_deliveries', [])
            reroute_type = serializer.validated_data.get('reroute_type', 'traffic')
            
            rerouting_service = ReroutingService()
            result_dto: Optional[OptimizationResult] = None 
            
            if reroute_type == 'traffic':
                traffic_data_input = serializer.validated_data.get('traffic_data', {}) # Default to empty dict
                traffic_data_for_service: Dict[Tuple[int, int], float] = {}
                
                if traffic_data_input: # traffic_data_input is already a dict from TrafficDataSerializer
                    location_id_to_idx = {loc.id: i for i, loc in enumerate(locations)}
                    if 'location_pairs' in traffic_data_input and 'factors' in traffic_data_input:
                        # ... (same conversion logic as in OptimizeRoutesView) ...
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
                            parts = key.split('-')
                            if len(parts) == 2:
                                from_idx = location_id_to_idx.get(parts[0])
                                to_idx = location_id_to_idx.get(parts[1])
                                if from_idx is not None and to_idx is not None:
                                    traffic_data_for_service[(from_idx, to_idx)] = float(factor)
                
                result_dto = rerouting_service.reroute_for_traffic(
                    current_routes=current_routes_dto,
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos,
                    completed_deliveries=completed_deliveries,
                    traffic_data=traffic_data_for_service
                )
            elif reroute_type == 'delay':
                delayed_location_ids = serializer.validated_data.get('delayed_location_ids', [])
                delay_minutes = serializer.validated_data.get('delay_minutes', {})
                result_dto = rerouting_service.reroute_for_delay(
                    current_routes=current_routes_dto, 
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos,
                    completed_deliveries=completed_deliveries,
                    delayed_location_ids=delayed_location_ids,
                    delay_minutes=delay_minutes
                )
            elif reroute_type == 'roadblock':
                blocked_segments_input = serializer.validated_data.get('blocked_segments', [])
                # blocked_segments in ReroutingRequestSerializer is List[List[str]]
                # ReroutingService.reroute_for_roadblock expects List[Tuple[str, str]]
                blocked_segments_tuples = [tuple(segment) for segment in blocked_segments_input]
                result_dto = rerouting_service.reroute_for_roadblock(
                    current_routes=current_routes_dto,
                    locations=locations,
                    vehicles=vehicles,
                    original_deliveries=original_deliveries_dtos,
                    completed_deliveries=completed_deliveries,
                    blocked_segments=blocked_segments_tuples
                )
            
            if result_dto:
                response_data = {
                    "status": result_dto.status,
                    "total_distance": result_dto.total_distance,
                    "total_cost": result_dto.total_cost,
                    "routes": result_dto.detailed_routes, # Map DTO's detailed_routes to serializer's routes
                    "unassigned_deliveries": result_dto.unassigned_deliveries,
                    "statistics": result_dto.statistics
                }
                response_serializer = RouteOptimizationResponseSerializer(data=response_data)
                if not response_serializer.is_valid():
                     logger.error(f"RerouteView response serialization error: {response_serializer.errors}")
                     return Response(response_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                return Response(response_serializer.data, status=status.HTTP_200_OK)
            else:
                # This case should ideally be handled by exceptions in ReroutingService returning an error DTO
                logger.error("Rerouting did not produce a result DTO for an unknown reason.")
                return Response({"error": "Invalid reroute type or no result obtained from rerouting service."}, status=status.HTTP_400_BAD_REQUEST)
            
        # Suggested change:
        except Exception as e:
            logger.exception("Critical error during rerouting: %s", str(e)) # Logger already captures the full str(e) and stack trace
            return Response(
                {"error": "An unexpected error occurred during rerouting. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@swagger_auto_schema(
    method='get',
    operation_id="health_check_get",
    operation_description="Performs a health check of the API. Returns the operational status of the service.",
    responses={
        200: openapi.Response(
            description="API is healthy and operational.",
            examples={"application/json": {"status": "healthy"}}
        ),
        503: openapi.Response(
            description="API is unhealthy or unavailable (example).",
            examples={"application/json": {"status": "unhealthy"}}
        )
    },
    tags=['Health Check']
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