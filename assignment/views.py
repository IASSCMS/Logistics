from django.utils import timezone
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models.assignment import Assignment
from .models.assignment_item import AssignmentItem
from fleet.models import Vehicle, VehicleLocation
from shipments.models import Shipment
from .serializers.assignment import AssignmentSerializer


class AssignmentViewSet(viewsets.ModelViewSet):
    queryset = Assignment.objects.all()
    serializer_class = AssignmentSerializer

    def create(self, request, *args, **kwargs):
        deliveries = request.data.get("deliveries")
        if not deliveries:
            return Response({"error": "Deliveries required"}, status=400)

        total_load = sum(d.get("load", 0) for d in deliveries)
        vehicle = Vehicle.objects.filter(status="available", capacity__gte=total_load).first()
        if not vehicle:
            return Response({"error": "No available vehicle for the load"}, status=400)

        vehicle.status = "assigned"
        vehicle.save()

        assignment = Assignment.objects.create(
            vehicle=vehicle,
            total_load=total_load,
            status='created'
        )

        for delivery in deliveries:
            shipment_id = delivery.get("shipment_id")
            location = delivery.get("location")
            sequence = delivery.get("sequence", 1)
            role = delivery.get("role")

            if role not in ["pickup", "delivery"]:
                return Response({"error": f"Invalid role for shipment {shipment_id}. Must be 'pickup' or 'delivery'."},
                                status=400)

            try:
                shipment = Shipment.objects.get(id=shipment_id)
            except Shipment.DoesNotExist:
                return Response({"error": f"Shipment {shipment_id} does not exist"}, status=400)

            AssignmentItem.objects.create(
                assignment=assignment,
                shipment=shipment,
                delivery_sequence=sequence,
                delivery_location=location,
                role=role
            )

        serializer = self.get_serializer(assignment)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["get"], url_path="by-vehicle/(?P<vehicle_id>[^/.]+)")
    def by_vehicle(self, request, vehicle_id=None):
        try:
            vehicle = Vehicle.objects.get(vehicle_id=vehicle_id)
        except Vehicle.DoesNotExist:
            return Response({"error": "Vehicle not found"}, status=404)

        assignment = Assignment.objects.filter(vehicle=vehicle).order_by('-id').first()
        if not assignment:
            return Response({"message": "No assignment found for this vehicle"}, status=404)

        serializer = self.get_serializer(assignment)
        return Response(serializer.data)

    @action(detail=True, methods=['post'], url_path='arrive/sequence/(?P<sequence>[0-9]+)')
    def mark_arrival(self, request, pk=None, sequence=None):
        assignment = self.get_object()
        vehicle = assignment.vehicle
        sequence = int(sequence)

        try:
            current_item = assignment.items.get(delivery_sequence=sequence)
        except AssignmentItem.DoesNotExist:
            return Response({"error": f"No assignment item found at sequence {sequence}"}, status=404)

        location = current_item.delivery_location
        lat, lng = location.get("lat"), location.get("lng")

        if lat is None or lng is None:
            return Response({"error": "Location data is missing in assignment item"}, status=400)

        vehicle.update_location(lat, lng)
        VehicleLocation.objects.create(
            vehicle=vehicle,
            latitude=lat,
            longitude=lng,
        )

        items_at_location = assignment.items.filter(
            delivery_location=location,
            delivery_sequence__gte=sequence
        ).order_by("delivery_sequence")

        grouped = [
            {
                "assignment_item_id": item.id,
                "role": item.role,
                "shipment_id": item.shipment.id,
                "shipment_status": item.shipment.status,
                "location": item.delivery_location,
                "is_delivered": item.is_delivered
            }
            for item in items_at_location
        ]

        return Response({
            "vehicle": vehicle.vehicle_id,
            "arrived_at": timezone.now(),
            "location": location,
            "actions": grouped
        })

    @action(detail=True, methods=["post"], url_path="actions/(?P<item_id>[0-9]+)/complete")
    def mark_action_complete(self, request, pk=None, item_id=None):
        try:
            assignment = self.get_object()
            item = assignment.items.get(id=item_id)
        except AssignmentItem.DoesNotExist:
            return Response({"error": "Assignment item not found"}, status=404)

        if item.is_delivered:
            return Response({"message": "Already marked complete"}, status=200)

        item.is_delivered = True
        item.delivered_at = timezone.now()
        item.save(update_fields=["is_delivered", "delivered_at"])

        # Optional: update shipment status
        if item.role == "delivery":
            item.shipment.mark_delivered()
        elif item.role == "pickup":
            item.shipment.mark_dispatched()
            item.shipment.mark_in_transit()
        item.shipment.save()

        return Response({
            "message": f"{item.role.title()} confirmed",
            "shipment_id": item.shipment.id,
            "new_status": item.shipment.status,
            "timestamp": item.delivered_at
        }, status=200)
