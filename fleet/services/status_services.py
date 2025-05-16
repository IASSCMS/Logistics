from django.utils import timezone
from fleet.models import Vehicle

def update_vehicle_status(vehicle: Vehicle, new_status: str):
    vehicle.status = new_status
    vehicle.updated_at = timezone.now()
    vehicle.save(update_fields=['status', 'updated_at'])

def mark_vehicle_available(vehicle: Vehicle):
    update_vehicle_status(vehicle, 'available')

def mark_vehicle_assigned(vehicle: Vehicle):
    update_vehicle_status(vehicle, 'assigned')

def mark_vehicle_maintenance(vehicle: Vehicle):
    update_vehicle_status(vehicle, 'maintenance')

def mark_vehicle_out_of_service(vehicle: Vehicle):
    update_vehicle_status(vehicle, 'out_of_service')
