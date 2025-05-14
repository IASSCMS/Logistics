import json
import unittest # For dataclass tests
from django.test import TestCase # For Django model tests
from django.db import IntegrityError
from django.utils import timezone
from datetime import timedelta
from django.test import TestCase # Import Django's TestCase
from django.db.utils import IntegrityError # For the uniqueness test

from route_optimizer.models import DistanceMatrixCache # Your model
from route_optimizer.models import Vehicle, Delivery, DistanceMatrixCache
from route_optimizer.core.constants import DEFAULT_DELIVERY_PRIORITY

class TestVehicleDataclass(unittest.TestCase):
    def test_vehicle_creation_all_fields(self):
        vehicle = Vehicle(
            id="V001",
            capacity=100.5,
            start_location_id="DEPOT_A",
            end_location_id="DEPOT_B",
            cost_per_km=1.5,
            fixed_cost=50.0,
            max_distance=500.0,
            max_stops=10,
            available=False,
            skills=["refrigeration", "express"]
        )
        self.assertEqual(vehicle.id, "V001")
        self.assertEqual(vehicle.capacity, 100.5)
        self.assertEqual(vehicle.start_location_id, "DEPOT_A")
        self.assertEqual(vehicle.end_location_id, "DEPOT_B")
        self.assertEqual(vehicle.cost_per_km, 1.5)
        self.assertEqual(vehicle.fixed_cost, 50.0)
        self.assertEqual(vehicle.max_distance, 500.0)
        self.assertEqual(vehicle.max_stops, 10)
        self.assertFalse(vehicle.available)
        self.assertEqual(vehicle.skills, ["refrigeration", "express"])

    def test_vehicle_creation_required_fields_and_defaults(self):
        vehicle = Vehicle(id="V002", capacity=75.0, start_location_id="DEPOT_C")
        self.assertEqual(vehicle.id, "V002")
        self.assertEqual(vehicle.capacity, 75.0)
        self.assertEqual(vehicle.start_location_id, "DEPOT_C")
        self.assertIsNone(vehicle.end_location_id) # Default
        self.assertEqual(vehicle.cost_per_km, 1.0) # Default
        self.assertEqual(vehicle.fixed_cost, 0.0) # Default
        self.assertIsNone(vehicle.max_distance) # Default
        self.assertIsNone(vehicle.max_stops) # Default
        self.assertTrue(vehicle.available) # Default
        self.assertEqual(vehicle.skills, []) # Default factory


class TestDeliveryDataclass(unittest.TestCase):
    def test_delivery_creation_all_fields(self):
        delivery = Delivery(
            id="D001",
            location_id="CUST001",
            demand=10.0,
            priority=3, # Higher than default
            required_skills=["fragile"],
            is_pickup=True
        )
        self.assertEqual(delivery.id, "D001")
        self.assertEqual(delivery.location_id, "CUST001")
        self.assertEqual(delivery.demand, 10.0)
        self.assertEqual(delivery.priority, 3)
        self.assertEqual(delivery.required_skills, ["fragile"])
        self.assertTrue(delivery.is_pickup)

    def test_delivery_creation_required_fields_and_defaults(self):
        delivery = Delivery(id="D002", location_id="CUST002", demand=5.0)
        self.assertEqual(delivery.id, "D002")
        self.assertEqual(delivery.location_id, "CUST002")
        self.assertEqual(delivery.demand, 5.0)
        self.assertEqual(delivery.priority, DEFAULT_DELIVERY_PRIORITY) # Default
        self.assertEqual(delivery.required_skills, []) # Default factory
        self.assertFalse(delivery.is_pickup) # Default


class TestDistanceMatrixCacheModel(TestCase):
    def test_create_distance_matrix_cache_entry(self):
        matrix_data_list = [[0, 10], [10, 0]]
        location_ids_list = ["loc1", "loc2"]
        time_matrix_data_list = [[0, 5], [5, 0]]

        entry = DistanceMatrixCache.objects.create(
            cache_key="test_key_123",
            matrix_data=json.dumps(matrix_data_list),
            location_ids=json.dumps(location_ids_list),
            time_matrix_data=json.dumps(time_matrix_data_list)
        )
        self.assertIsNotNone(entry.pk)
        self.assertEqual(entry.cache_key, "test_key_123")
        self.assertEqual(json.loads(entry.matrix_data), matrix_data_list)
        self.assertEqual(json.loads(entry.location_ids), location_ids_list)
        self.assertEqual(json.loads(entry.time_matrix_data), time_matrix_data_list)
        self.assertIsNotNone(entry.created_at)
        self.assertTrue(timezone.now() - entry.created_at < timedelta(minutes=1))

    def test_cache_key_uniqueness(self):
        DistanceMatrixCache.objects.create(
            cache_key="unique_key_test",
            matrix_data="[[0]]",
            location_ids="[\"locA\"]"
            # time_matrix_data can be None by default
        )
        with self.assertRaises(IntegrityError):
            DistanceMatrixCache.objects.create(
                cache_key="unique_key_test", # Same key
                matrix_data="[[1]]",
                location_ids="[\"locB\"]"
            )

    def test_time_matrix_data_nullable(self):
        entry = DistanceMatrixCache.objects.create(
            cache_key="test_key_no_time",
            matrix_data="[[0, 10], [10, 0]]",
            location_ids="[\"loc1\", \"loc2\"]",
            time_matrix_data=None # Explicitly None
        )
        self.assertIsNone(entry.time_matrix_data)

        entry_blank = DistanceMatrixCache.objects.create(
            cache_key="test_key_blank_time",
            matrix_data="[[0, 10], [10, 0]]",
            location_ids="[\"loc1\", \"loc2\"]",
            time_matrix_data="" # TextField(null=True, blank=True) stores "" as ""
        )
        self.assertEqual(entry_blank.time_matrix_data, "")

    def test_verbose_names(self):
        self.assertEqual(DistanceMatrixCache._meta.verbose_name, "Distance Matrix Cache")
        self.assertEqual(DistanceMatrixCache._meta.verbose_name_plural, "Distance Matrix Caches")

    def test_model_indexes(self):
        # Check if indexes are defined
        # This is more of a check that the definition exists, not that the DB created them correctly in a unit test
        # Database-level index checks are usually part of integration tests or DB schema inspection
        index_names = [index.name for index in DistanceMatrixCache._meta.indexes]
        self.assertIn('route_optim_cache_k_8e6d1d_idx', index_names) # Name from migration 0001
        self.assertIn('route_optim_created_087bbe_idx', index_names) # Name from migration 0001


if __name__ == '__main__':
    # This allows running unittest for dataclasses directly if needed,
    # but for Django model tests, 'python manage.py test' is preferred.
    unittest.main()

