import os
import django

from fleet.services.status_services import mark_vehicle_assigned, mark_vehicle_available, update_vehicle_status

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'logistics_core.settings')
django.setup()

from django.db.models import Sum, Count
from django.utils import timezone
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from datetime import datetime

from fleet.models import Vehicle, VehicleLocation
from django.conf import settings

if settings.ENABLE_FLEET_EXTENDED_MODELS:
    from fleet.models import MaintenanceRecord

from fleet.serializers import VehicleSerializer, VehicleDetailSerializer


class VehicleViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing vehicles.
    """
    queryset = Vehicle.objects.all()
    serializer_class = VehicleSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['status', 'fuel_type', 'depot_id']
    search_fields = ['vehicle_id', 'name', 'plate_number', 'depot_id']
    ordering_fields = ['vehicle_id', 'capacity', 'status', 'created_at']
    ordering = ['vehicle_id']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return VehicleDetailSerializer
        return super().get_serializer_class()

    def get_queryset(self):
        queryset = super().get_queryset()
        params = self.request.query_params

        if status := params.get('status'):
            queryset = queryset.filter(status=status)
        if min_cap := params.get('min_capacity'):
            try:
                queryset = queryset.filter(capacity__gte=int(min_cap))
            except ValueError:
                pass
        if max_cap := params.get('max_capacity'):
            try:
                queryset = queryset.filter(capacity__lte=int(max_cap))
            except ValueError:
                pass
        if params.get('available') == 'true':
            queryset = queryset.filter(status='available')
        if depot := params.get('depot_id'):
            queryset = queryset.filter(depot_id=depot)
        queryset = queryset.order_by('-updated_at')
        return queryset

    @action(detail=True, methods=['post'])
    def mark_available(self, request, pk=None):
        vehicle = self.get_object()
        mark_vehicle_available(vehicle)
        return Response({'vehicle_id': vehicle.vehicle_id, 'status': 'available'})

    @action(detail=True, methods=['post'])
    def mark_assigned(self, request, pk=None):
        vehicle = self.get_object()
        mark_vehicle_assigned(vehicle)
        return Response({'vehicle_id': vehicle.vehicle_id, 'status': 'assigned'})

    # Admin only
    @action(detail=True, methods=['post'])
    def change_status(self, request, pk=None):
        vehicle = self.get_object()
        new_status = request.data.get('status')

        valid_statuses = dict(Vehicle.STATUS_CHOICES).keys()
        if new_status not in valid_statuses:
            return Response({'error': f'Invalid status. Must be one of {list(valid_statuses)}'}, status=400)

        update_vehicle_status(vehicle, new_status)
        return Response({'vehicle_id': vehicle.vehicle_id, 'status': new_status})

    @action(detail=True, methods=['post'])
    def update_location(self, request, pk=None):
        vehicle = self.get_object()
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        speed = request.data.get('speed')
        heading = request.data.get('heading')

        if latitude is None or longitude is None:
            return Response({'error': 'Latitude and longitude are required'}, status=400)

        try:
            vehicle.update_location(latitude, longitude)
            VehicleLocation.objects.create(
                vehicle=vehicle,
                latitude=latitude,
                longitude=longitude,
                speed=speed or None,
                heading=heading or None
            )
            return Response({'status': 'location updated'}, status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=400)

    @action(detail=True, methods=['post'])
    def assign_depot(self, request, pk=None):
        """
        Assign or update a vehicle's depot.
        POST /api/fleet/vehicles/{id}/assign_depot/
        {
            "depot_id": "WHS001",
            "latitude": 6.9271,
            "longitude": 79.8612
        }
        """
        vehicle = self.get_object()
        depot_id = request.data.get('depot_id')
        depot_lat = request.data.get('latitude')
        depot_lon = request.data.get('longitude')

        if depot_id is None:
            return Response({'error': 'depot_id is required'}, status=400)

        vehicle.depot_id = depot_id

        if depot_lat is not None and depot_lon is not None:
            try:
                vehicle.depot_latitude = float(depot_lat)
                vehicle.depot_longitude = float(depot_lon)
            except ValueError:
                return Response({'error': 'Invalid latitude or longitude'}, status=400)

        vehicle.save(update_fields=['depot_id', 'depot_latitude', 'depot_longitude', 'updated_at'])
        return Response(VehicleSerializer(vehicle).data)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        status_counts = dict(
            Vehicle.objects.values('status').annotate(count=Count('id')).values_list('status', 'count')
        )
        for s, _ in Vehicle.STATUS_CHOICES:
            status_counts.setdefault(s, 0)

        total_vehicles = Vehicle.objects.count()
        total_capacity = Vehicle.objects.aggregate(Sum('capacity'))['capacity__sum'] or 0
        available_capacity = Vehicle.objects.filter(status='available').aggregate(Sum('capacity'))['capacity__sum'] or 0
        maintenance_count = 0

        if settings.ENABLE_FLEET_EXTENDED_MODELS:
            maintenance_count = MaintenanceRecord.objects.filter(status__in=['scheduled', 'in_progress']).count()

        utilization_rate = 0
        if total_vehicles:
            utilization_rate = (status_counts.get('assigned', 0) / total_vehicles) * 100

        return Response({
            'total_vehicles': total_vehicles,
            'status_counts': status_counts,
            'total_capacity': total_capacity,
            'available_capacity': available_capacity,
            'maintenance_count': maintenance_count,
            'utilization_rate': utilization_rate
        })

    @action(detail=False, methods=['get'])
    def by_depot(self, request):
        depot_id = request.query_params.get('depot_id')
        if not depot_id:
            return Response({'error': 'Missing depot_id parameter'}, status=400)

        vehicles = Vehicle.objects.filter(depot_id=depot_id)
        return Response(VehicleSerializer(vehicles, many=True).data)

    @action(detail=False, methods=['get'])
    def depot_stats(self, request):
        """
        Returns count and total capacity of vehicles per depot.
        """
        stats = Vehicle.objects.values('depot_id').annotate(
            count=Count('id'),
            total_capacity=Sum('capacity')
        ).order_by('depot_id')

        return Response({'by_depot': stats})

    # # To be implemented with maintenance part
    # @action(detail=True, methods=['post'])
    # def change_status(self, request, pk=None):
    #     vehicle = self.get_object()
    #     new_status = request.data.get('status')
    #
    #     if not new_status:
    #         return Response({'error': 'Status is required'}, status=400)
    #
    #     if new_status not in dict(Vehicle.STATUS_CHOICES):
    #         return Response({'error': f'Invalid status: {new_status}'}, status=400)
    #
    #     if new_status == 'maintenance' and vehicle.status != 'maintenance' and settings.ENABLE_FLEET_EXTENDED_MODELS:
    #         maintenance_type = request.data.get('maintenance_type', 'routine')
    #         description = request.data.get('description', 'Routine maintenance')
    #         scheduled_date = request.data.get('scheduled_date', timezone.now().date().isoformat())
    #         try:
    #             scheduled_date = datetime.fromisoformat(scheduled_date).date()
    #         except ValueError:
    #             scheduled_date = timezone.now().date()
    #
    #         MaintenanceRecord.objects.create(
    #             vehicle=vehicle,
    #             maintenance_type=maintenance_type,
    #             description=description,
    #             scheduled_date=scheduled_date,
    #             status='in_progress'
    #         )
    #
    #     vehicle.status = new_status
    #     vehicle.save(update_fields=['status', 'updated_at'])
    #     return Response(VehicleSerializer(vehicle).data)