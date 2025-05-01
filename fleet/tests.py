from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework import status
import json
from datetime import datetime, timedelta

from .models import (
    Vehicle, MaintenanceRecord, FuelRecord, 
    TripRecord, VehicleLocation
)

class VehicleModelTest(TestCase):
    """Test vehicle model functionality."""
    
    def setUp(self):
        self.vehicle = Vehicle.objects.create(
            vehicle_id="TRK001",
            name="Test Truck 1",
            capacity=1000,
            status="available",
            fuel_type="diesel",
            plate_number="ABC123",
            year_of_manufacture=2020,
            fuel_efficiency=8.5
        )
    
    def test_vehicle_creation(self):
        """Test that vehicle can be created."""
        self.assertEqual(self.vehicle.vehicle_id, "TRK001")
        self.assertEqual(self.vehicle.capacity, 1000)
        self.assertEqual(self.vehicle.status, "available")
        self.assertTrue(self.vehicle.is_available)
    
    def test_update_location(self):
        """Test updating vehicle location."""
        # Initial location should be None
        self.assertIsNone(self.vehicle.current_latitude)
        self.assertIsNone(self.vehicle.current_longitude)
        
        # Update location
        latitude = 45.123456
        longitude = -75.654321
        
        self.vehicle.update_location(latitude, longitude)
        
        # Check that location was updated
        self.assertEqual(float(self.vehicle.current_latitude), latitude)
        self.assertEqual(float(self.vehicle.current_longitude), longitude)
        self.assertIsNotNone(self.vehicle.last_location_update)
        
        # Check that location isn't stale right after update
        self.assertFalse(self.vehicle.location_is_stale)


class MaintenanceRecordModelTest(TestCase):
    """Test maintenance record functionality."""
    
    def setUp(self):
        self.vehicle = Vehicle.objects.create(
            vehicle_id="TRK002",
            capacity=1500,
            status="available"
        )
        
        self.maintenance = MaintenanceRecord.objects.create(
            vehicle=self.vehicle,
            maintenance_type="routine",
            status="scheduled",
            description="Regular oil change",
            scheduled_date=timezone.now().date() + timedelta(days=5)
        )
    
    def test_complete_maintenance(self):
        """Test completing maintenance."""
        # Set vehicle to maintenance status
        self.vehicle.status = "maintenance"
        self.vehicle.save()
        
        # Complete maintenance
        completion_date = timezone.now().date()
        cost = 150.75
        
        self.maintenance.complete_maintenance(completion_date, cost)
        
        # Check that maintenance is completed
        self.maintenance.refresh_from_db()
        self.assertEqual(self.maintenance.status, "completed")
        self.assertEqual(self.maintenance.completion_date, completion_date)
        self.assertEqual(float(self.maintenance.cost), cost)
        
        # Check that vehicle status was updated
        self.vehicle.refresh_from_db()
        self.assertEqual(self.vehicle.status, "available")


class VehicleAPITest(TestCase):
    """Test vehicle API endpoints."""
    
    def setUp(self):
        self.client = APIClient()
        self.vehicle1 = Vehicle.objects.create(
            vehicle_id="TRK001", capacity=1000, status="available", 
            name="Truck 1", fuel_type="diesel"
        )
        self.vehicle2 = Vehicle.objects.create(
            vehicle_id="TRK002", capacity=500, status="maintenance", 
            name="Truck 2", fuel_type="petrol"
        )
        self.vehicle3 = Vehicle.objects.create(
            vehicle_id="TRK003", capacity=750, status="assigned", 
            name="Truck 3", fuel_type="diesel"
        )
    
    def test_list_all_vehicles(self):
        """Test retrieving all vehicles."""
        response = self.client.get('/api/fleet/vehicles/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 3)
    
    def test_filter_by_status(self):
        """Test filtering vehicles by status."""
        response = self.client.get('/api/fleet/vehicles/?status=available')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['vehicle_id'], 'TRK001')
    
    def test_filter_by_min_capacity(self):
        """Test filtering vehicles by minimum capacity."""
        response = self.client.get('/api/fleet/vehicles/?min_capacity=800')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['vehicle_id'], 'TRK001')
    
    def test_filter_by_fuel_type(self):
        """Test filtering vehicles by fuel type."""
        response = self.client.get('/api/fleet/vehicles/?fuel_type=diesel')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
        vehicle_ids = [v['vehicle_id'] for v in response.data]
        self.assertIn('TRK001', vehicle_ids)
        self.assertIn('TRK003', vehicle_ids)
    
    def test_create_vehicle(self):
        """Test creating a new vehicle."""
        payload = {
            'vehicle_id': 'TRK004',
            'name': 'Truck 4',
            'capacity': 1200,
            'status': 'available',
            'fuel_type': 'electric',
            'plate_number': 'XYZ789'
        }
        response = self.client.post('/api/fleet/vehicles/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['vehicle_id'], 'TRK004')
        self.assertEqual(response.data['fuel_type'], 'electric')
    
    def test_update_vehicle(self):
        """Test updating an existing vehicle."""
        response = self.client.patch(
            f'/api/fleet/vehicles/{self.vehicle1.id}/', 
            {'status': 'maintenance'}, 
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'maintenance')
    
    def test_update_location(self):
        """Test updating vehicle location."""
        payload = {
            'latitude': 42.123456,
            'longitude': -71.654321,
            'speed': 65.5
        }
        response = self.client.post(
            f'/api/fleet/vehicles/{self.vehicle1.id}/update_location/', 
            payload, 
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Check that location was updated in the vehicle
        self.vehicle1.refresh_from_db()
        self.assertEqual(float(self.vehicle1.current_latitude), 42.123456)
        self.assertEqual(float(self.vehicle1.current_longitude), -71.654321)
        
        # Check that a location history record was created
        location_history = VehicleLocation.objects.filter(vehicle=self.vehicle1)
        self.assertEqual(location_history.count(), 1)
        self.assertEqual(float(location_history[0].speed), 65.5)


class MaintenanceAPITest(TestCase):
    """Test maintenance API endpoints."""
    
    def setUp(self):
        self.client = APIClient()
        self.vehicle = Vehicle.objects.create(
            vehicle_id="TRK001", capacity=1000, status="available"
        )
        self.maintenance = MaintenanceRecord.objects.create(
            vehicle=self.vehicle,
            maintenance_type="routine",
            status="scheduled",
            description="Oil change and inspection",
            scheduled_date=timezone.now().date() + timedelta(days=3)
        )
    
    def test_list_maintenance_records(self):
        """Test retrieving all maintenance records."""
        response = self.client.get('/api/fleet/maintenance/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
    
    def test_create_maintenance_record(self):
        """Test creating a new maintenance record."""
        scheduled_date = (timezone.now().date() + timedelta(days=5)).isoformat()
        payload = {
            'vehicle': self.vehicle.id,
            'maintenance_type': 'repair',
            'status': 'scheduled',
            'description': 'Brake replacement',
            'scheduled_date': scheduled_date
        }
        response = self.client.post('/api/fleet/maintenance/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['maintenance_type'], 'repair')
        self.assertEqual(response.data['status'], 'scheduled')
    
    def test_complete_maintenance(self):
        """Test completing a maintenance record."""
        completion_date = timezone.now().date().isoformat()
        payload = {
            'completion_date': completion_date,
            'cost': 250.75
        }
        response = self.client.post(
            f'/api/fleet/maintenance/{self.maintenance.id}/complete/', 
            payload, 
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'completed')
        self.assertEqual(response.data['completion_date'], completion_date)
        self.assertEqual(float(response.data['cost']), 250.75)


class FuelRecordAPITest(TestCase):
    """Test fuel record API endpoints."""
    
    def setUp(self):
        self.client = APIClient()
        self.vehicle = Vehicle.objects.create(
            vehicle_id="TRK001", capacity=1000, status="available",
            fuel_type="diesel", fuel_efficiency=8.5
        )
    
    def test_create_fuel_record(self):
        """Test creating a new fuel record."""
        payload = {
            'vehicle': self.vehicle.id,
            'refuel_date': timezone.now().isoformat(),
            'amount': 75.5,
            'cost': 120.25,
            'odometer_reading': 5000,
            'location_name': 'Gas Station ABC'
        }
        response = self.client.post('/api/fleet/fuel/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(float(response.data['amount']), 75.5)
        self.assertEqual(float(response.data['cost']), 120.25)
        self.assertEqual(response.data['odometer_reading'], 5000)


class TripRecordAPITest(TestCase):
    """Test trip record API endpoints."""
    
    def setUp(self):
        self.client = APIClient()
        self.vehicle = Vehicle.objects.create(
            vehicle_id="TRK001", capacity=1000, status="available"
        )
        
        start_time = timezone.now() - timedelta(hours=2)
        self.trip = TripRecord.objects.create(
            vehicle=self.vehicle,
            start_time=start_time,
            start_odometer=5000,
            driver_name="Test Driver",
            purpose="Delivery to Warehouse A"
        )
    
    def test_create_trip_record(self):
        """Test creating a new trip record."""
        start_time = (timezone.now() - timedelta(hours=1)).isoformat()
        payload = {
            'vehicle': self.vehicle.id,
            'start_time': start_time,
            'start_odometer': 5500,
            'driver_name': 'Another Driver',
            'purpose': 'Pickup from Supplier B'
        }
        response = self.client.post('/api/fleet/trips/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['start_odometer'], 5500)
        self.assertEqual(response.data['driver_name'], 'Another Driver')
    
    def test_end_trip(self):
        """Test ending a trip."""
        end_time = timezone.now().isoformat()
        payload = {
            'end_time': end_time,
            'end_odometer': 5150,
            'end_latitude': 40.123456,
            'end_longitude': -74.654321,
            'notes': 'Trip completed successfully'
        }
        response = self.client.post(
            f'/api/fleet/trips/{self.trip.id}/end_trip/', 
            payload, 
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Check that trip was updated
        self.trip.refresh_from_db()
        self.assertEqual(self.trip.end_odometer, 5150)
        self.assertEqual(float(self.trip.end_latitude), 40.123456)
        self.assertEqual(float(self.trip.end_longitude), -74.654321)
        
        # Verify calculated properties
        self.assertEqual(self.trip.distance, 150)  # 5150 - 5000