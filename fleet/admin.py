from django.contrib import admin
from django.conf import settings
from .models import Vehicle, VehicleLocation

@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = (
        'vehicle_id', 'name', 'capacity', 'status', 'fuel_type',
        'depot_id', 'depot_latitude', 'depot_longitude', 'last_location_update'
    )
    list_filter = ('status', 'fuel_type')
    search_fields = ('vehicle_id', 'name', 'plate_number', 'depot_id')
    readonly_fields = ('created_at', 'updated_at', 'last_location_update')

    fieldsets = (
        ('Basic Information', {
            'fields': (
                'vehicle_id', 'name', 'plate_number',
                'year_of_manufacture', 'status'
            )
        }),
        ('Depot Assignment', {
            'fields': ('depot_id', 'depot_latitude', 'depot_longitude')
        }),
        ('Specifications', {
            'fields': ('capacity', 'fuel_type', 'max_speed', 'fuel_efficiency')
        }),
        ('Current Location', {
            'fields': ('current_latitude', 'current_longitude', 'last_location_update')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(VehicleLocation)
class VehicleLocationAdmin(admin.ModelAdmin):
    list_display = ('vehicle', 'timestamp', 'latitude', 'longitude', 'speed', 'heading')
    list_filter = ('timestamp',)
    search_fields = ('vehicle__vehicle_id',)
    raw_id_fields = ('vehicle',)
    date_hierarchy = 'timestamp'

# Conditionally register extended models
if getattr(settings, 'ENABLE_FLEET_EXTENDED_MODELS', False):
    from .models import MaintenanceRecord, FuelRecord, TripRecord

    @admin.register(MaintenanceRecord)
    class MaintenanceRecordAdmin(admin.ModelAdmin):
        list_display = ('vehicle', 'maintenance_type', 'status', 'scheduled_date', 'completion_date')
        list_filter = ('maintenance_type', 'status', 'scheduled_date')
        search_fields = ('vehicle__vehicle_id', 'description')
        readonly_fields = ('created_at', 'updated_at')
        raw_id_fields = ('vehicle',)
        date_hierarchy = 'scheduled_date'

    @admin.register(FuelRecord)
    class FuelRecordAdmin(admin.ModelAdmin):
        list_display = ('vehicle', 'refuel_date', 'amount', 'cost', 'odometer_reading')
        list_filter = ('refuel_date',)
        search_fields = ('vehicle__vehicle_id', 'location_name', 'notes')
        readonly_fields = ('created_at',)
        raw_id_fields = ('vehicle',)
        date_hierarchy = 'refuel_date'

    @admin.register(TripRecord)
    class TripRecordAdmin(admin.ModelAdmin):
        list_display = ('vehicle', 'start_time', 'end_time', 'distance', 'driver_name')
        list_filter = ('start_time',)
        search_fields = ('vehicle__vehicle_id', 'driver_name', 'purpose', 'notes')
        readonly_fields = ('created_at', 'updated_at', 'distance', 'duration')
        raw_id_fields = ('vehicle',)
        date_hierarchy = 'start_time'
        fieldsets = (
            ('Trip Information', {
                'fields': ('vehicle', 'driver_name', 'purpose', 'notes')
            }),
            ('Timing', {
                'fields': ('start_time', 'end_time', 'duration')
            }),
            ('Distance', {
                'fields': ('start_odometer', 'end_odometer', 'distance')
            }),
            ('Location', {
                'fields': (
                    'start_latitude', 'start_longitude',
                    'end_latitude', 'end_longitude'
                )
            }),
            ('Timestamps', {
                'fields': ('created_at', 'updated_at'),
                'classes': ('collapse',)
            })
        )
