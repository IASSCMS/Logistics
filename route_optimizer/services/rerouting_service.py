"""
Service for dynamic rerouting based on real-time events.

This module provides functionality to dynamically adjust routes
based on unexpected events like traffic, delays, or roadblocks.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
import copy
import numpy as np

from route_optimizer.core.distance_matrix import Location, DistanceMatrixBuilder
from route_optimizer.core.ortools_optimizer import Vehicle, Delivery
from route_optimizer.services.optimization_service import OptimizationService

# Set up logging
logger = logging.getLogger(__name__)


class ReroutingService:
    """
    Service for dynamic rerouting of vehicles based on real-time events.
    """
    
    def __init__(self, optimization_service: Optional[OptimizationService] = None):
        """
        Initialize the rerouting service.
        
        Args:
            optimization_service: Optimization service to use for rerouting.
                                 If None, a new service will be created.
        """
        self.optimization_service = optimization_service or OptimizationService()
    
    def reroute_for_traffic(
        self,
        current_routes: Dict[str, Any],
        locations: List[Location],
        vehicles: List[Vehicle],
        completed_deliveries: List[str],
        traffic_data: Dict[Tuple[int, int], float]
    ) -> Dict[str, Any]:
        """
        Reroute vehicles based on updated traffic data.
        
        Args:
            current_routes: Current route plan.
            locations: List of all locations.
            vehicles: List of all vehicles.
            completed_deliveries: IDs of deliveries that have been completed.
            traffic_data: Dictionary mapping (location_idx, location_idx) to traffic factors.
                         A factor > 1.0 means slower traffic.
        
        Returns:
            Updated route plan.
        """
        # Filter out completed deliveries
        remaining_deliveries = self._get_remaining_deliveries(
            current_routes, completed_deliveries
        )
        
        # Update vehicle positions
        updated_vehicles = self._update_vehicle_positions(
            vehicles, current_routes, completed_deliveries
        )
        
        # Re-optimize with traffic data
        new_routes = self.optimization_service.optimize_routes(
            locations=locations,
            vehicles=updated_vehicles,
            deliveries=remaining_deliveries,
            consider_traffic=True,
            traffic_data=traffic_data
        )
        
        # Add rerouting metadata
        new_routes['rerouting_info'] = {
            'reason': 'traffic',
            'traffic_factors': len(traffic_data),
            'completed_deliveries': len(completed_deliveries),
            'remaining_deliveries': len(remaining_deliveries)
        }
        
        return new_routes
    
    def reroute_for_delay(
        self,
        current_routes: Dict[str, Any],
        locations: List[Location],
        vehicles: List[Vehicle],
        completed_deliveries: List[str],
        delayed_location_ids: List[str],
        delay_minutes: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Reroute vehicles based on loading/unloading delays.
        
        Args:
            current_routes: Current route plan.
            locations: List of all locations.
            vehicles: List of all vehicles.
            completed_deliveries: IDs of deliveries that have been completed.
            delayed_location_ids: IDs of locations experiencing delays.
            delay_minutes: Dictionary mapping location IDs to delay in minutes.
        
        Returns:
            Updated route plan.
        """
        # Update service times for delayed locations
        updated_locations = copy.deepcopy(locations)
        for location in updated_locations:
            if location.id in delayed_location_ids:
                # Add delay to service time
                location.service_time += delay_minutes.get(location.id, 0)
        
        # Filter out completed deliveries
        remaining_deliveries = self._get_remaining_deliveries(
            current_routes, completed_deliveries
        )
        
        # Update vehicle positions
        updated_vehicles = self._update_vehicle_positions(
            vehicles, current_routes, completed_deliveries
        )
        
        # Re-optimize with updated service times
        new_routes = self.optimization_service.optimize_routes(
            locations=updated_locations,
            vehicles=updated_vehicles,
            deliveries=remaining_deliveries,
            consider_time_windows=True
        )
        
        # Add rerouting metadata
        new_routes['rerouting_info'] = {
            'reason': 'service_delay',
            'delayed_locations': len(delayed_location_ids),
            'completed_deliveries': len(completed_deliveries),
            'remaining_deliveries': len(remaining_deliveries)
        }
        
        return new_routes
    
    def reroute_for_roadblock(
        self,
        current_routes: Dict[str, Any],
        locations: List[Location],
        vehicles: List[Vehicle],
        completed_deliveries: List[str],
        blocked_segments: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Reroute vehicles based on road blocks.
        
        Args:
            current_routes: Current route plan.
            locations: List of all locations.
            vehicles: List of all vehicles.
            completed_deliveries: IDs of deliveries that have been completed.
            blocked_segments: List of tuples (location_id1, location_id2) representing blocked roads.
        
        Returns:
            Updated route plan.
        """
        # Create distance matrix
        distance_matrix, location_ids = DistanceMatrixBuilder.create_distance_matrix(
            locations, use_haversine=True
        )
        
        # Create location ID to index mapping
        location_id_to_index = {loc_id: i for i, loc_id in enumerate(location_ids)}
        
        # Apply roadblocks by setting distances to infinity
        for from_id, to_id in blocked_segments:
            try:
                from_idx = location_id_to_index[from_id]
                to_idx = location_id_to_index[to_id]
                
                # Set both directions to infinity (very high value)
                distance_matrix[from_idx, to_idx] = float('inf')
                distance_matrix[to_idx, from_idx] = float('inf')
            except KeyError:
                logger.warning(f"Location ID not found when applying roadblock: {from_id} or {to_id}")
        
        # Filter out completed deliveries
        remaining_deliveries = self._get_remaining_deliveries(
            current_routes, completed_deliveries
        )
        
        # Update vehicle positions
        updated_vehicles = self._update_vehicle_positions(
            vehicles, current_routes, completed_deliveries
        )
        
        # Create a custom traffic data structure for the modified distances
        traffic_data = {}
        for i in range(len(location_ids)):
            for j in range(len(location_ids)):
                if distance_matrix[i, j] == float('inf'):
                    traffic_data[(i, j)] = float('inf')
        
        # Re-optimize with roadblock data
        new_routes = self.optimization_service.optimize_routes(
            locations=locations,
            vehicles=updated_vehicles,
            deliveries=remaining_deliveries,
            consider_traffic=True,
            traffic_data=traffic_data
        )
        
        # Add rerouting metadata
        new_routes['rerouting_info'] = {
            'reason': 'roadblock',
            'blocked_segments': len(blocked_segments),
            'completed_deliveries': len(completed_deliveries),
            'remaining_deliveries': len(remaining_deliveries)
        }
        
        return new_routes
    
    def _get_remaining_deliveries(
        self,
        current_routes: Dict[str, Any],
        completed_delivery_ids: List[str]
    ) -> List[Delivery]:
        """
        Extract the remaining deliveries that need to be completed.
        
        Args:
            current_routes: Current route plan.
            completed_delivery_ids: IDs of deliveries that have been completed.
        
        Returns:
            List of remaining Delivery objects.
        """
        # This is a placeholder implementation
        # In a real system, you'd need to map from the route structure to actual Delivery objects
        # For this example, we'll assume the system stores deliveries in current_routes['deliveries']
        
        completed_set = set(completed_delivery_ids)
        remaining_deliveries = []
        
        if 'deliveries' in current_routes:
            for delivery in current_routes['deliveries']:
                if delivery.id not in completed_set:
                    remaining_deliveries.append(delivery)
        
        return remaining_deliveries
    
    def _update_vehicle_positions(
        self,
        vehicles: List[Vehicle],
        current_routes: Dict[str, Any],
        completed_delivery_ids: List[str]
    ) -> List[Vehicle]:
        """
        Update vehicle positions based on completed deliveries.
        
        Args:
            vehicles: List of original Vehicle objects.
            current_routes: Current route plan.
            completed_delivery_ids: IDs of deliveries that have been completed.
        
        Returns:
            Updated list of Vehicle objects with new start locations.
        """
        # Create a mapping from delivery IDs to location IDs
        delivery_to_location = {}
        if 'deliveries' in current_routes:
            for delivery in current_routes['deliveries']:
                delivery_to_location[delivery.id] = delivery.location_id
        
        # Create a deep copy of vehicles to modify
        updated_vehicles = copy.deepcopy(vehicles)
        
        # Map vehicle IDs to their assigned routes
        vehicle_routes = {}
        if 'assigned_vehicles' in current_routes:
            for vehicle_id, route_idx in current_routes['assigned_vehicles'].items():
                if 'detailed_routes' in current_routes and route_idx < len(current_routes['detailed_routes']):
                    vehicle_routes[vehicle_id] = current_routes['detailed_routes'][route_idx]['stops']
        
        # Update each vehicle's starting position
        for vehicle in updated_vehicles:
            route = vehicle_routes.get(vehicle.id)
            if not route:
                continue
            
            # Find the last completed delivery for this vehicle
            last_completed_idx = -1
            for i, location_id in enumerate(route):
                # Check if any completed delivery is at this location
                for delivery_id in completed_delivery_ids:
                    if delivery_to_location.get(delivery_id) == location_id:
                        last_completed_idx = max(last_completed_idx, i)
            
            # Update vehicle starting location if deliveries have been completed
            if last_completed_idx >= 0 and last_completed_idx < len(route) - 1:
                # Set new starting location to the location after the last completed one
                vehicle.start_location_id = route[last_completed_idx + 1]
        
        return updated_vehicles