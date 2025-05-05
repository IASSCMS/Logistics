from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import Assignment
from .serializers import AssignmentSerializer
from fleet.models import Vehicle
from route_optimizer.optimizer import RouteOptimizer
from django.conf import settings
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AssignmentViewSet(viewsets.ModelViewSet):
    queryset = Assignment.objects.all()
    serializer_class = AssignmentSerializer

    def create(self, request, *args, **kwargs):
        # Log the raw payload for debugging
        logger.debug("Received payload: %s", request.data)

        # Check for deliveries
        deliveries = request.data.get("deliveries")
        if not deliveries:
            logger.error("No deliveries provided")
            return Response({"error": "Deliveries required"}, status=400)

        # Validate deliveries format
        if not isinstance(deliveries, list):
            logger.error("Deliveries must be a list")
            return Response({"error": "Deliveries must be a list"}, status=400)

        for delivery in deliveries:
            location = delivery.get("location")
            load = delivery.get("load")
            if not isinstance(location, list) or len(location) != 2:
                logger.error("Invalid location format: %s", delivery)
                return Response({"error": f"Invalid location format for delivery: {delivery}"}, status=400)
            if not all(isinstance(coord, (int, float)) for coord in location):
                logger.error("Location coordinates must be numbers: %s", delivery)
                return Response({"error": f"Location coordinates must be numbers: {delivery}"}, status=400)
            if not isinstance(load, (int, float)) or load <= 0:
                logger.error("Invalid load: %s", delivery)
                return Response({"error": f"Invalid load for delivery: {delivery}"}, status=400)

        # Calculate total load
        total_load = sum(d.get("load", 0) for d in deliveries)
        logger.debug("Total load: %s", total_load)

        # Find an available vehicle
        vehicle = Vehicle.objects.filter(status="available", capacity__gte=total_load).first()
        if not vehicle:
            logger.error("No available vehicle for load: %s", total_load)
            return Response({"error": "No available vehicle for the load"}, status=400)
        logger.debug("Selected vehicle: %s", vehicle.vehicle_id)

        # Prepare data for RouteOptimizer
        depot = getattr(settings, 'DEPOT_COORDINATES', [77.58, 12.96])
        locations = [
            {'id': 'depot', 'coordinates': depot, 'load': 0}
        ] + [
            {
                'id': f'loc_{i}',
                'coordinates': d.get("location"),
                'load': d.get("load")
            }
            for i, d in enumerate(deliveries)
        ]
        logger.debug("RouteOptimizer locations: %s", locations)

        # Run RouteOptimizer
        try:
            optimizer = RouteOptimizer(locations, [vehicle.capacity])
            routes = optimizer.solve()
            if not routes:
                logger.error("No feasible route found for locations: %s", locations)
                return Response({"error": "No feasible route found"}, status=400)

            # Extract the optimized route
            optimized_route = routes[0]['route']
            optimized_distance = routes[0]['distance']
            logger.debug("Optimized route: %s, distance: %s", optimized_route, optimized_distance)

            # Map optimized route to coordinates, excluding depot
            location_map = {loc['id']: loc['coordinates'] for loc in locations}
            delivery_locations = [location_map[loc_id] for loc_id in optimized_route[1:-1]]

        except ValueError as e:
            logger.error("RouteOptimizer validation error: %s", str(e))
            return Response({"error": f"Invalid input for route optimization: {str(e)}"}, status=400)
        except Exception as e:
            logger.error("RouteOptimizer error: %s", str(e))
            return Response({"error": f"Route optimization failed: {str(e)}"}, status=500)

        # Mark vehicle as assigned
        vehicle.status = "assigned"
        vehicle.save()

        # Create Assignment
        assignment = Assignment.objects.create(
            vehicle=vehicle,
            delivery_locations=delivery_locations,
            total_load=total_load,
            optimized_distance=optimized_distance
        )
        logger.debug("Created assignment: %s", assignment)

        serializer = self.get_serializer(assignment)
        return Response(serializer.data, status=status.HTTP_201_CREATED)