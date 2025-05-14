from rest_framework import serializers
from fleet.models import Vehicle, VehicleLocation


class VehicleSerializer(serializers.ModelSerializer):
    location_is_stale = serializers.BooleanField(read_only=True)
    is_available = serializers.BooleanField(read_only=True)

    class Meta:
        ref_name = 'FleetVehicle'
        model = Vehicle
        fields = [
            'id', 'vehicle_id', 'name', 'capacity', 'status', 'fuel_type',
            'plate_number', 'year_of_manufacture', 'current_latitude',
            'current_longitude', 'last_location_update', 'max_speed',
            'fuel_efficiency', 'created_at', 'updated_at', 'is_available',
            'location_is_stale'
        ]
        read_only_fields = ['created_at', 'updated_at', 'last_location_update']

class VehicleLocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = VehicleLocation
        fields = ['timestamp', 'latitude', 'longitude', 'speed', 'heading']
