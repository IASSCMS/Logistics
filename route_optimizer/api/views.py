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

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.services.rerouting_service import ReroutingService
from route_optimizer.core.distance_matrix import Location
from route_optimizer.core.ortools_optimizer import Vehicle, Delivery
from route_optimizer.api.serializers import (
    RouteOptimizationRequestSerializer,
    RouteOptimizationResponseSerializer,
    ReroutingRequestSerializer,
    LocationSerializer,
    VehicleSerializer,
    DeliverySerializer
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
            # Convert serialized data to domain objects
            locations = [
                Location(
                    id=loc_data['id'],
                    name=loc_data['name'],
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    address=loc_data.get('address'),
                    is_depot=loc_data.get('is_depot', False),
                    time_window_start=loc_data.get('time_window_start'),
                    time_window_end=loc_data.get('time_window_end'),
                    service_time=loc_data.get('service_time', 15)
                )
                for loc_data in serializer.validated_data['locations']
            ]
            
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
                )
                for veh_data in serializer.validated_data['vehicles']
            ]
            
            deliveries = [
                Delivery(
                    id=del_data['id'],
                    location_id=del_data['location_id'],
                    demand=del_data['demand'],
                    priority=del_data.get('priority', 1),
                    required_skills=del_data.get('required_skills', []),
                    is_pickup=del_data.get('is_pickup', False)
                )
                for del_data in serializer.validated_data['deliveries']
            ]
            
            consider_traffic = serializer.validated_data.get('consider_traffic', False)
            consider_time_windows = serializer.validated_data.get('consider_time_windows', False)
            
            # Call the optimization service
            optimization_service = OptimizationService()
            result = optimization_service.optimize_routes(
                locations=locations,
                vehicles=vehicles,
                deliveries=deliveries,
                consider_traffic=consider_traffic,
                consider_time_windows=consider_time_windows
            )
            
            # Return the result
            response_serializer = RouteOptimizationResponseSerializer(result)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception("Error during route optimization: %s", str(e))
            return Response(
                {"error": f"Route optimization failed: {str(e)}"},
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
            # Convert serialized data to domain objects
            locations = [
                Location(
                    id=loc_data['id'],
                    name=loc_data['name'],
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    address=loc_data.get('address'),
                    is_depot=loc_data.get('is_depot', False),
                    time_window_start=loc_data.get('time_window_start'),
                    time_window_end=loc_data.get('time_window_end'),
                    service_time=loc_data.get('service_time', 15)
                )
                for loc_data in serializer.validated_data['locations']
            ]
            
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
                )
                for veh_data in serializer.validated_data['vehicles']
            ]
            
            current_routes = serializer.validated_data['current_routes']
            completed_deliveries = serializer.validated_data.get('completed_deliveries', [])
            reroute_type = serializer.validated_data.get('reroute_type', 'traffic')
            
            # Create the rerouting service
            rerouting_service = ReroutingService()
            result = None
            
            # Call the appropriate rerouting method based on the type
            if reroute_type == 'traffic':
                traffic_data_list = serializer.validated_data.get('traffic_data', {})
                traffic_data = {}
                
                # Convert traffic data from list format to dictionary
                if traffic_data_list:
                    for i, pair in enumerate(traffic_data_list.get('location_pairs', [])):
                        if i < len(traffic_data_list.get('factors', [])):
                            # Find indices of the locations
                            from_idx = next((i for i, loc in enumerate(locations) if loc.id == pair[0]), None)
                            to_idx = next((i for i, loc in enumerate(locations) if loc.id == pair[1]), None)
                            
                            if from_idx is not None and to_idx is not None:
                                traffic_data[(from_idx, to_idx)] = traffic_data_list['factors'][i]
                
                result = rerouting_service.reroute_for_traffic(
                    current_routes=current_routes,
                    locations=locations,
                    vehicles=vehicles,
                    completed_deliveries=completed_deliveries,
                    traffic_data=traffic_data
                )
                
            elif reroute_type == 'delay':
                delayed_location_ids = serializer.validated_data.get('delayed_location_ids', [])
                delay_minutes = serializer.validated_data.get('delay_minutes', {})
                
                result = rerouting_service.reroute_for_delay(
                    current_routes=current_routes,
                    locations=locations,
                    vehicles=vehicles,
                    completed_deliveries=completed_deliveries,
                    delayed_location_ids=delayed_location_ids,
                    delay_minutes=delay_minutes
                )
                
            elif reroute_type == 'roadblock':
                blocked_segments = [tuple(segment) for segment in serializer.validated_data.get('blocked_segments', [])]
                
                result = rerouting_service.reroute_for_roadblock(
                    current_routes=current_routes,
                    locations=locations,
                    vehicles=vehicles,
                    completed_deliveries=completed_deliveries,
                    blocked_segments=blocked_segments
                )
            
            # Return the result
            response_serializer = RouteOptimizationResponseSerializer(result)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
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