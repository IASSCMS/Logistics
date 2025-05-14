"""
Service for dynamic rerouting based on real-time events.

This module provides functionality to dynamically adjust routes
based on unexpected events like traffic, delays, or roadblocks.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import copy

from route_optimizer.core.distance_matrix import DistanceMatrixBuilder
from route_optimizer.core.types_1 import Location, OptimizationResult, ReroutingInfo,  validate_optimization_result
from route_optimizer.models import Vehicle, Delivery
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
        current_routes: OptimizationResult,
        locations: List[Location],
        vehicles: List[Vehicle],
        original_deliveries: List[Delivery],
        completed_deliveries: List[str],
        traffic_data: Dict[Tuple[int, int], float]
    ) -> OptimizationResult:
        """
        Reroute vehicles based on traffic conditions.

        Args:
            current_routes: Current OptimizationResult with route plans.
            locations: List of Location objects.
            vehicles: List of Vehicle objects.
            completed_deliveries: List of delivery IDs that have been completed.
            traffic_data: Dictionary mapping (from_id, to_id) pairs to traffic factors.
                        A factor > 1.0 means slower traffic.

        Returns:
            OptimizationResult with updated routes accounting for traffic conditions.
            The statistics field will contain rerouting_info with details about the rerouting.
        """
        try:
            # Filter out completed deliveries
            remaining_deliveries = self._get_remaining_deliveries(
                original_deliveries, completed_deliveries
            )
            
            # Update vehicle positions
            updated_vehicles = self._update_vehicle_positions(
                vehicles, current_routes, completed_deliveries, original_deliveries
            )
            
            # Re-optimize with traffic data
            new_routes = self.optimization_service.optimize_routes(
                locations=locations,
                vehicles=updated_vehicles,
                deliveries=remaining_deliveries,
                consider_traffic=True,
                traffic_data=traffic_data
            )
            
            # Create ReroutingInfo DTO, Assuming optimize_routes now consistently returns OptimizationResult DTO
            rerouting_info = ReroutingInfo(
                reason='traffic',
                traffic_factors=len(traffic_data), # Count of distinct traffic segments affected
                completed_deliveries=len(completed_deliveries),
                remaining_deliveries=len(remaining_deliveries)
            )
            # Add to statistics
            if not new_routes.statistics: # Ensure statistics dict exists
                new_routes.statistics = {}
            new_routes.statistics['rerouting_info'] = vars(rerouting_info)
            
            return new_routes
        
        except Exception as e:
            logger.error(f"Error during reroute_for_traffic: {str(e)}", exc_info=True)
            return OptimizationResult(
                status='error',
                unassigned_deliveries=[d.id for d in original_deliveries] if original_deliveries else [],
                statistics={'error': f"Rerouting for traffic failed: {str(e)}"}
            )
    
    def reroute_for_delay(
        self,
        current_routes: OptimizationResult,
        locations: List[Location],
        vehicles: List[Vehicle],
        original_deliveries: List[Delivery],
        completed_deliveries: List[str],
        delayed_location_ids: List[str],
        delay_minutes: Dict[str, int]
    ) -> OptimizationResult:
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
        try:
            # Update service times for delayed locations
            updated_locations = copy.deepcopy(locations)
            for location in updated_locations:
                if location.id in delayed_location_ids:
                    # Add delay to service time
                    location.service_time += delay_minutes.get(location.id, 0)
            
            # Filter out completed deliveries
            remaining_deliveries = self._get_remaining_deliveries(
                original_deliveries, completed_deliveries
            )
            
            # Update vehicle positions
            updated_vehicles = self._update_vehicle_positions(
                vehicles, current_routes, completed_deliveries, original_deliveries
            )
            
            # Re-optimize with updated service times
            new_routes = self.optimization_service.optimize_routes(
                locations=updated_locations,
                vehicles=updated_vehicles,
                deliveries=remaining_deliveries,
                consider_time_windows=True # Delays are most impactful with time windows
            )
            
            # Create ReroutingInfo DTO
            rerouting_info = ReroutingInfo(
                reason='service_delay',
                delay_locations=delayed_location_ids, # ReroutingInfo.delay_locations is List[str]
                completed_deliveries=len(completed_deliveries),
                remaining_deliveries=len(remaining_deliveries)
            )
            # Add to statistics
            if not new_routes.statistics:
                new_routes.statistics = {}
            new_routes.statistics['rerouting_info'] = vars(rerouting_info)
            
            return new_routes
        
        except Exception as e:
            logger.error(f"Error during reroute_for_delay: {str(e)}", exc_info=True)
            return OptimizationResult(
                status='error',
                unassigned_deliveries=[d.id for d in original_deliveries] if original_deliveries else [],
                statistics={'error': f"Rerouting for delay failed: {str(e)}"}
            )

    
    def reroute_for_roadblock(
        self,
        current_routes: OptimizationResult,
        locations: List[Location],
        vehicles: List[Vehicle],
        original_deliveries: List[Delivery],
        completed_deliveries: List[str],
        blocked_segments: List[Tuple[str, str]]
    ) -> OptimizationResult:
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
        try:
            # For roadblocks, we primarily modify the distance/cost aspect.
            # The non-API path of create_distance_matrix should provide what we need here.
            # Create distance matrix
            matrix_data = DistanceMatrixBuilder.create_distance_matrix(
                locations, use_haversine=True, average_speed_kmh=None # To get (dist_km, None, loc_ids)
            )
            distance_matrix, _, location_ids = matrix_data # Unpack, ignoring the time matrix part
            
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
                original_deliveries, completed_deliveries
            )
            
            # Update vehicle positions
            updated_vehicles = self._update_vehicle_positions(
                vehicles, current_routes, completed_deliveries, original_deliveries
            )
            
            # Semantic Note: Using 'traffic_data' for roadblocks is a practical way to make
            # segments unusable by assigning them infinite cost/time.
            # This relies on OptimizationService._apply_traffic_safely to handle 'inf'
            # or very large numbers appropriately, or for the VRP solver to interpret them.
            # Create a custom traffic data structure for the modified distances
            traffic_data_for_roadblocks = {}
            for r_idx in range(len(location_ids)):
                for c_idx in range(len(location_ids)):
                    if distance_matrix[r_idx, c_idx] == float('inf'):
                        traffic_data_for_roadblocks[(r_idx, c_idx)] = float('inf') 
            
            # Re-optimize with roadblock data
            new_routes = self.optimization_service.optimize_routes(
                locations=locations,
                vehicles=updated_vehicles,
                deliveries=remaining_deliveries,
                consider_traffic=True,
                traffic_data=traffic_data_for_roadblocks
            )
            
            rerouting_info_dto = ReroutingInfo(
                reason='roadblock',
                # Note: ReroutingInfo DTO has 'delay_locations' and 'traffic_factors',
                # but not directly 'blocked_segments' as an int. You might want to add
                # 'blocked_segments_count' to ReroutingInfo DTO or use an existing field.
                # For now, let's assume we want to store the count in a generic way
                # or adapt the DTO. For simplicity, I'll put it in statistics directly
                # if not fitting neatly into ReroutingInfo.
                blocked_segments=blocked_segments, # Pass the actual list of tuples, Stores the list of (from_id, to_id) tuples
                completed_deliveries=len(completed_deliveries),
                remaining_deliveries=len(remaining_deliveries)
            )
            if not new_routes.statistics:
                new_routes.statistics = {}
                    
            # Add specific roadblock info
            new_routes.statistics['rerouting_info'] = vars(rerouting_info_dto)
            # Add count for convenience, consistent with ReroutingInfo DTO having the list
            new_routes.statistics['rerouting_info']['blocked_segments_count'] = len(blocked_segments)
            
            return new_routes
        
        except Exception as e:
            logger.error(f"Error during reroute_for_roadblock: {str(e)}", exc_info=True)
            return OptimizationResult(
                status='error',
                unassigned_deliveries=[d.id for d in original_deliveries] if original_deliveries else [],
                statistics={'error': f"Rerouting for roadblock failed: {str(e)}"}
            )
    
    def _get_remaining_deliveries(
        self,
        original_deliveries: List[Delivery],
        completed_delivery_ids: List[str]
    ) -> List[Delivery]:
        """
        Extract the remaining deliveries from the original list that need to be completed.
        
        Args:
            original_deliveries: The full list of Delivery objects that were initially planned.
            completed_delivery_ids: IDs of deliveries that have been completed.
        
        Returns:
            List of remaining Delivery objects.
        """
        completed_set = set(completed_delivery_ids)
        remaining_deliveries = [
            delivery for delivery in original_deliveries if delivery.id not in completed_set
        ]
        
        if not original_deliveries and completed_delivery_ids:
            logger.warning("_get_remaining_deliveries: Original deliveries list is empty, but completed IDs were provided.")
        elif not original_deliveries:
            logger.info("_get_remaining_deliveries: Original deliveries list is empty.")

        return remaining_deliveries
    
    def _update_vehicle_positions(
        self,
        vehicles: List[Vehicle],
        current_routes: OptimizationResult,
        completed_delivery_ids: List[str],
        original_deliveries: List[Delivery]
    ) -> List[Vehicle]:
        """
        Update vehicle positions based on completed deliveries.
        This is a simplified approach: assumes vehicle is at the *next planned stop*
        after its last completed delivery in the *current_routes* plan.
        
        Args:
            vehicles: List of original Vehicle objects.
            current_routes: Dictionary representation of the current route plan.
            completed_delivery_ids: IDs of deliveries that have been completed.
            original_deliveries: The full list of Delivery objects to map delivery IDs to locations.
        
        Returns:
            Updated list of Vehicle objects with new start locations.
        """
        delivery_to_location = {
            delivery.id: delivery.location_id for delivery in original_deliveries
        }
        
        updated_vehicles = copy.deepcopy(vehicles)
        
        # Access attributes directly from OptimizationResult DTO
        assigned_vehicles_map = current_routes.assigned_vehicles 
        detailed_routes_list = current_routes.detailed_routes   # This is List[Dict[str, Any]]

        for vehicle in updated_vehicles:
            route_idx = assigned_vehicles_map.get(vehicle.id) # assigned_vehicles is a Dict
            
            if route_idx is None or not detailed_routes_list or route_idx >= len(detailed_routes_list):
                logger.debug(f"Vehicle {vehicle.id} not in current_routes.assigned_vehicles or route index invalid.")
                continue # Vehicle not in current plan or route index invalid

            current_vehicle_route_info = detailed_routes_list[route_idx]
            # Ensure 'stops' are actual location IDs, not indices
            route_stops = current_vehicle_route_info.get('stops', []) 
            
            if not route_stops:
                logger.debug(f"No stops found for vehicle {vehicle.id} in its detailed route.")
                continue

            last_completed_stop_index_in_route = -1
            for i, stop_location_id in enumerate(route_stops):
                # Check if any completed delivery corresponds to this stop_location_id
                for completed_id in completed_delivery_ids:
                    if delivery_to_location.get(completed_id) == stop_location_id:
                        # This stop had a completed delivery. Mark its index.
                        last_completed_stop_index_in_route = max(last_completed_stop_index_in_route, i)
            
            # If a delivery was completed on this route and it's not the very last stop
            if 0 <= last_completed_stop_index_in_route < len(route_stops) - 1:
                # New start location is the stop *after* the last completed one
                new_start_location_id = route_stops[last_completed_stop_index_in_route + 1]
                logger.info(f"Updating vehicle {vehicle.id} start location from {vehicle.start_location_id} to {new_start_location_id} based on completed deliveries.")
                vehicle.start_location_id = new_start_location_id
            elif last_completed_stop_index_in_route == len(route_stops) -1:
                 # All stops on this route completed, vehicle is at its planned end.
                 # Keep its original end_location_id or start if end is not defined.
                 # This part might need more sophisticated logic if vehicle should be "free"
                 logger.info(f"Vehicle {vehicle.id} completed all stops on its route. Positioned at planned end {route_stops[last_completed_stop_index_in_route]}.")
                 vehicle.start_location_id = route_stops[last_completed_stop_index_in_route]
            # else: No completed deliveries on this route, or already at the next stop. No change.


        return updated_vehicles