"""
Tests for the optimization service.

This module contains tests for the OptimizationService class.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.core.ortools_optimizer import Vehicle, Delivery
from route_optimizer.core.distance_matrix import Location, DistanceMatrixBuilder


class TestOptimizationService(unittest.TestCase):
    """Test cases for OptimizationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = OptimizationService()
        
        # Sample locations
        self.locations = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True),
            Location(id="customer1", name="Customer 1", latitude=1.0, longitude=0.0),
            Location(id="customer2", name="Customer 2", latitude=0.0, longitude=1.0),
            Location(id="customer3", name="Customer 3", latitude=1.0, longitude=1.0)
        ]
        
        # Sample vehicles
        self.vehicles = [
            Vehicle(
                id="vehicle1",
                capacity=10.0,
                start_location_id="depot",
                end_location_id="depot"
            ),
            Vehicle(
                id="vehicle2",
                capacity=15.0,
                start_location_id="depot",
                end_location_id="depot"
            )
        ]
        
        # Sample deliveries
        self.deliveries = [
            Delivery(id="delivery1", location_id="customer1", demand=5.0),
            Delivery(id="delivery2", location_id="customer2", demand=3.0),
            Delivery(id="delivery3", location_id="customer3", demand=6.0)
        ]
        
        # Mock distance matrix and location IDs
        self.distance_matrix = np.array([
            [0.0, 1.0, 1.0, 1.4],
            [1.0, 0.0, 1.4, 1.0],
            [1.0, 1.4, 0.0, 1.0],
            [1.4, 1.0, 1.0, 0.0]
        ])
        self.location_ids = ["depot", "customer1", "customer2", "customer3"]

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_routes_basic(self, mock_solve, mock_create_matrix):
        """Test basic route optimization without traffic or time windows."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_solve.return_value = {
            'status': 'success',
            'routes': [[0, 1, 2, 0], [0, 3, 0]],
            'total_distance': 6.0,
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1},
            'unassigned_deliveries': []
        }
        
        # Call the service
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        
        # Verify the result
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['total_distance'], 6.0)
        self.assertEqual(len(result['routes']), 2)
        self.assertEqual(len(result['unassigned_deliveries']), 0)
        
        # Verify the mocks were called correctly
        mock_create_matrix.assert_called_once()
        mock_solve.assert_called_once()

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.add_traffic_factors')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_routes_with_traffic(self, mock_solve, mock_add_traffic, mock_create_matrix):
        """Test route optimization with traffic data."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_add_traffic.return_value = self.distance_matrix  # Just return the same matrix for simplicity
        mock_solve.return_value = {
            'status': 'success',
            'routes': [[0, 1, 2, 0], [0, 3, 0]],
            'total_distance': 6.0,
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1},
            'unassigned_deliveries': []
        }
        
        # Sample traffic data
        traffic_data = {(0, 1): 1.5, (1, 2): 1.2}
        
        # Call the service
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            consider_traffic=True,
            traffic_data=traffic_data
        )
        
        # Verify the result
        self.assertEqual(result['status'], 'success')
        
        # Verify the mocks were called correctly
        mock_create_matrix.assert_called_once()
        mock_add_traffic.assert_called_once_with(self.distance_matrix, traffic_data)
        mock_solve.assert_called_once()

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve_with_time_windows')
    def test_optimize_routes_with_time_windows(self, mock_solve_tw, mock_create_matrix):
        """Test route optimization with time windows."""
        # Set up mocks
        mock_create_matrix.return_value = (self.distance_matrix, self.location_ids)
        mock_solve_tw.return_value = {
            'status': 'success',
            'routes': [[0, 1, 2, 0], [0, 3, 0]],
            'total_distance': 6.0,
            'assigned_vehicles': {'vehicle1': 0, 'vehicle2': 1},
            'unassigned_deliveries': [],
            'arrival_times': {0: [0, 20, 40], 1: [0, 30]}
        }
        
        # Create locations with time windows
        locations_with_tw = [
            Location(
                id="depot", 
                name="Depot", 
                latitude=0.0, 
                longitude=0.0, 
                is_depot=True,
                time_window_start=0,
                time_window_end=1440
            ),
            Location(
                id="customer1", 
                name="Customer 1", 
                latitude=1.0, 
                longitude=0.0,
                time_window_start=480,
                time_window_end=600
            ),
            Location(
                id="customer2", 
                name="Customer 2", 
                latitude=0.0, 
                longitude=1.0,
                time_window_start=540,
                time_window_end=660
            ),
            Location(
                id="customer3", 
                name="Customer 3", 
                latitude=1.0, 
                longitude=1.0,
                time_window_start=600,
                time_window_end=720
            )
        ]
        
        # Call the service
        result = self.service.optimize_routes(
            locations=locations_with_tw,
            vehicles=self.vehicles,
            deliveries=self.deliveries,
            consider_time_windows=True
        )
        
        # Verify the result
        self.assertEqual(result['status'], 'success')
        self.assertIn('arrival_times', result)
        
        # Verify the mocks were called correctly
        mock_create_matrix.assert_called_once()
        mock_solve_tw.assert_called_once()
