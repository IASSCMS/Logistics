import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project_name.settings')
django.setup()
from django.db.models import QuerySet, Q, Sum, Count, Avg
from django.utils import timezone
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from datetime import datetime, timedelta

from .models import (
    Vehicle, MaintenanceRecord, FuelRecord, 
    TripRecord, VehicleLocation
)
from .serializers import (
    VehicleSerializer, VehicleDetailSerializer,
    MaintenanceRecordSerializer, FuelRecordSerializer,
    TripRecordSerializer, VehicleLocationSerializer,
    MaintenanceScheduleSerializer
)

class VehicleViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing vehicles.
    """
    queryset = Vehicle.objects.all()
    serializer_class = VehicleSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['status', 'fuel_type']
    search_fields = ['vehicle_id', 'name', 'plate_number']
    ordering_fields = ['vehicle_id', 'capacity', 'status', 'created_at']
    ordering = ['vehicle_id']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return VehicleDetailSerializer
        return super().get_serializer_class()

    def get_queryset(self):
        queryset = super().get_queryset()
        request = self.request  # DRF Request, safe to use .query_params

        status = request.query_params.get('status')
        min_capacity = request.query_params.get('min_capacity')
        max_capacity = request.query_params.get('max_capacity')
        available_only = request.query_params.get('available') == 'true'

        if status:
            queryset = queryset.filter(status=status)
        if min_capacity:
            try:
                min_capacity = int(min_capacity)
                queryset = queryset.filter(capacity__gte=min_capacity)
            except ValueError:
                pass  # Ignore invalid capacity filters
        if max_capacity:
            try:
                max_capacity = int(max_capacity)
                queryset = queryset.filter(capacity__lte=max_capacity)
            except ValueError:
                pass
        if available_only:
            queryset = queryset.filter(status='available')

        return queryset
    
    @action(detail=True, methods=['post'])
    def update_location(self, request, pk=None):
        """
        Update vehicle location.
        POST /api/fleet/vehicles/{id}/update_location/
        """
        vehicle = self.get_object()
        
        # Extract location data from request
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        speed = request.data.get('speed')
        heading = request.data.get('heading')
        
        # Validate required fields
        if not latitude or not longitude:
            return Response(
                {'error': 'Latitude and longitude are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Update current location on vehicle
            vehicle.update_location(latitude, longitude)
            
            # Create a location history record
            location_data = {
                'vehicle': vehicle,
                'latitude': latitude,
                'longitude': longitude
            }
            
            if speed is not None:
                location_data['speed'] = speed
            if heading is not None:
                location_data['heading'] = heading
                
            VehicleLocation.objects.create(**location_data)
            
            return Response({'status': 'location updated'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['post'])
    def change_status(self, request, pk=None):
        """
        Change vehicle status.
        POST /api/fleet/vehicles/{id}/change_status/
        """
        vehicle = self.get_object()
        new_status = request.data.get('status')
        
        if not new_status:
            return Response(
                {'error': 'Status is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate status choice
        if new_status not in dict(Vehicle.STATUS_CHOICES):
            return Response(
                {'error': f'Invalid status. Must be one of: {dict(Vehicle.STATUS_CHOICES).keys()}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Handle status change to maintenance
        if new_status == 'maintenance' and vehicle.status != 'maintenance':
            # Optionally create a maintenance record
            maintenance_type = request.data.get('maintenance_type', 'routine')
            description = request.data.get('description', 'Routine maintenance')
            scheduled_date = request.data.get('scheduled_date', timezone.now().date().isoformat())
            
            try:
                scheduled_date = datetime.fromisoformat(scheduled_date).date()
            except ValueError:
                scheduled_date = timezone.now().date()
            
            # Create maintenance record
            MaintenanceRecord.objects.create(
                vehicle=vehicle,
                maintenance_type=maintenance_type,
                description=description,
                scheduled_date=scheduled_date,
                status='in_progress'  # Since we're changing status to maintenance now
            )
        
        # Update vehicle status
        vehicle.status = new_status
        vehicle.save(update_fields=['status', 'updated_at'])
        
        return Response(VehicleSerializer(vehicle).data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get fleet statistics.
        GET /api/fleet/vehicles/stats/
        """
        # Count vehicles by status
        status_counts = dict(
            Vehicle.objects.values('status').annotate(count=Count('id')).values_list('status', 'count')
        )
        
        # Fill in missing statuses with 0
        for status, _ in Vehicle.STATUS_CHOICES:
            if status not in status_counts:
                status_counts[status] = 0
        
        # Count total vehicles
        total_vehicles = Vehicle.objects.count()
        
        # Calculate total fleet capacity
        total_capacity = Vehicle.objects.aggregate(Sum('capacity'))['capacity__sum'] or 0
        
        # Calculate available capacity
        available_capacity = Vehicle.objects.filter(status='available').aggregate(
            Sum('capacity')
        )['capacity__sum'] or 0
        
        # Calculate maintenance stats
        maintenance_count = MaintenanceRecord.objects.filter(
            status__in=['scheduled', 'in_progress']
        ).count()
        
        # Current utilization rate
        utilization_rate = 0
        if total_vehicles > 0:
            assigned_count = status_counts.get('assigned', 0)
            utilization_rate = (assigned_count / total_vehicles) * 100
        
        return Response({
            'total_vehicles': total_vehicles,
            'status_counts': status_counts,
            'total_capacity': total_capacity,
            'available_capacity': available_capacity,
            'maintenance_count': maintenance_count,
            'utilization_rate': utilization_rate
        })


class MaintenanceRecordViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing maintenance records.
    """
    queryset = MaintenanceRecord.objects.all()
    serializer_class = MaintenanceRecordSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['vehicle', 'maintenance_type', 'status']
    ordering_fields = ['scheduled_date', 'completion_date', 'created_at']
    ordering = ['-scheduled_date']
    
    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """
        Mark maintenance as completed.
        POST /api/fleet/maintenance/{id}/complete/
        """
        maintenance = self.get_object()
        
        if maintenance.status == 'completed':
            return Response(
                {'error': 'This maintenance is already completed'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        completion_date = request.data.get('completion_date')
        cost = request.data.get('cost')
        
        try:
            if completion_date:
                completion_date = datetime.fromisoformat(completion_date).date()
            
            if cost:
                cost = float(cost)
                
            maintenance.complete_maintenance(completion_date, cost)
            return Response(
                MaintenanceRecordSerializer(maintenance).data,
                status=status.HTTP_200_OK
            )
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=False, methods=['get'])
    def upcoming(self, request):
        """
        List upcoming maintenance events.
        GET /api/fleet/maintenance/upcoming/
        """
        upcoming = MaintenanceRecord.objects.filter(
            status__in=['scheduled', 'in_progress'],
            scheduled_date__gte=timezone.now().date()
        ).order_by('scheduled_date')
        
        # Optionally filter by days in the future
        days = request.query_params.get('days')
        if days:
            try:
                days = int(days)
                future_date = timezone.now().date() + timedelta(days=days)
                upcoming = upcoming.filter(scheduled_date__lte=future_date)
            except ValueError:
                pass
        
        serializer = MaintenanceScheduleSerializer(upcoming, many=True)
        return Response(serializer.data)


class FuelRecordViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing fuel records.
    """
    queryset = FuelRecord.objects.all()
    serializer_class = FuelRecordSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['vehicle']
    ordering_fields = ['refuel_date', 'created_at']
    ordering = ['-refuel_date']
    
    @action(detail=False, methods=['get'])
    def consumption_stats(self, request):
        """
        Get fuel consumption statistics.
        GET /api/fleet/fuel/consumption_stats/
        """
        # Filter for specific vehicle if requested
        vehicle_id = request.query_params.get('vehicle_id')
        days = request.query_params.get('days', 30)
        
        try:
            days = int(days)
            start_date = timezone.now() - timedelta(days=days)
        except ValueError:
            days = 30
            start_date = timezone.now() - timedelta(days=days)
        
        queryset = FuelRecord.objects.filter(refuel_date__gte=start_date)
        
        if vehicle_id:
            queryset = queryset.filter(vehicle__vehicle_id=vehicle_id)
        
        # Calculate total cost and amount for the period
        totals = queryset.aggregate(
            total_cost=Sum('cost'),
            total_amount=Sum('amount')
        )
        
        # Get stats by vehicle
        vehicle_stats = []
        if not vehicle_id:
            # Group by vehicle and aggregate
            vehicles = Vehicle.objects.filter(
                fuel_records__refuel_date__gte=start_date
            ).distinct()
            
            for vehicle in vehicles:
                vehicle_records = queryset.filter(vehicle=vehicle)
                
                vehicle_totals = vehicle_records.aggregate(
                    total_cost=Sum('cost'),
                    total_amount=Sum('amount'),
                    count=Count('id')
                )
                
                vehicle_stats.append({
                    'vehicle_id': vehicle.vehicle_id,
                    'name': vehicle.name,
                    'records_count': vehicle_totals['count'],
                    'total_cost': vehicle_totals['total_cost'],
                    'total_amount': vehicle_totals['total_amount']
                })
        
        return Response({
            'period_days': days,
            'total_cost': totals['total_cost'] or 0,
            'total_amount': totals['total_amount'] or 0,
            'vehicle_stats': vehicle_stats
        })


class TripRecordViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing trip records.
    """
    queryset = TripRecord.objects.all()
    serializer_class = TripRecordSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['vehicle']
    ordering_fields = ['start_time', 'created_at']
    ordering = ['-start_time']
    
    @action(detail=True, methods=['post'])
    def end_trip(self, request, pk=None):
        """
        End a trip by setting end time and odometer.
        POST /api/fleet/trips/{id}/end_trip/
        """
        trip = self.get_object()
        
        if trip.end_time:
            return Response(
                {'error': 'This trip has already ended'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Extract end trip data
        end_time = request.data.get('end_time')
        end_odometer = request.data.get('end_odometer')
        end_latitude = request.data.get('end_latitude')
        end_longitude = request.data.get('end_longitude')
        notes = request.data.get('notes')
        
        # Validate required fields
        if not end_time or not end_odometer:
            return Response(
                {'error': 'End time and end odometer are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Parse end_time if it's a string
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            
            end_odometer = int(end_odometer)
            
            # Ensure end odometer is greater than start odometer
            if end_odometer <= trip.start_odometer:
                return Response(
                    {'error': 'End odometer must be greater than start odometer'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Update trip record
            trip.end_time = end_time
            trip.end_odometer = end_odometer
            
            if end_latitude and end_longitude:
                trip.end_latitude = end_latitude
                trip.end_longitude = end_longitude
                
            if notes:
                if trip.notes:
                    trip.notes += f"\n\nEnd trip notes: {notes}"
                else:
                    trip.notes = notes
            
            trip.save()
            
            # If we have coordinates, update vehicle location too
            if end_latitude and end_longitude:
                trip.vehicle.update_location(end_latitude, end_longitude)
            
            return Response(TripRecordSerializer(trip).data)
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get trip statistics.
        GET /api/fleet/trips/stats/
        """
        # Filter parameters
        vehicle_id = request.query_params.get('vehicle_id')
        days = request.query_params.get('days', 30)
        
        try:
            days = int(days)
            start_date = timezone.now() - timedelta(days=days)
        except ValueError:
            days = 30
            start_date = timezone.now() - timedelta(days=days)
        
        # Base queryset
        queryset = TripRecord.objects.filter(
            start_time__gte=start_date,
            end_time__isnull=False  # Only include completed trips
        )
        
        if vehicle_id:
            queryset = queryset.filter(vehicle__vehicle_id=vehicle_id)
        
        # Calculate total distance and duration
        total_distance = 0
        total_duration = 0
        
        # We need to iterate through the records to use the properties
        for trip in queryset:
            if trip.distance:
                total_distance += trip.distance
            if trip.duration:
                total_duration += trip.duration
        
        # Count total trips
        total_trips = queryset.count()
        
        # Average trip length and duration
        avg_distance = total_distance / total_trips if total_trips > 0 else 0
        avg_duration = total_duration / total_trips if total_trips > 0 else 0
        
        # Vehicle-specific stats if not filtering for a specific vehicle
        vehicle_stats = []
        if not vehicle_id:
            vehicles = Vehicle.objects.filter(
                trip_records__start_time__gte=start_date,
                trip_records__end_time__isnull=False
            ).distinct()
            
            for vehicle in vehicles:
                vehicle_trips = queryset.filter(vehicle=vehicle)
                
                v_total_distance = 0
                v_total_duration = 0
                
                for trip in vehicle_trips:
                    if trip.distance:
                        v_total_distance += trip.distance
                    if trip.duration:
                        v_total_duration += trip.duration
                
                v_trip_count = vehicle_trips.count()
                
                vehicle_stats.append({
                    'vehicle_id': vehicle.vehicle_id,
                    'name': vehicle.name,
                    'trip_count': v_trip_count,
                    'total_distance': v_total_distance,
                    'total_duration': v_total_duration,
                    'avg_distance': v_total_distance / v_trip_count if v_trip_count > 0 else 0,
                    'avg_duration': v_total_duration / v_trip_count if v_trip_count > 0 else 0
                })
        
        return Response({
            'period_days': days,
            'total_trips': total_trips,
            'total_distance': total_distance,
            'total_duration': total_duration,
            'avg_distance': avg_distance,
            'avg_duration': avg_duration,
            'vehicle_stats': vehicle_stats
        })