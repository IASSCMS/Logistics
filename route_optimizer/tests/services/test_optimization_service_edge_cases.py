import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from route_optimizer.services.optimization_service import OptimizationService
from route_optimizer.core.ortools_optimizer import Vehicle, Delivery
from route_optimizer.core.distance_matrix import Location

class OptimizationServiceEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.service = OptimizationService()
        self.locations = [
            Location(id="depot", name="Depot", latitude=0.0, longitude=0.0, is_depot=True),
        ]
        self.vehicles = [
            Vehicle(id="vehicle1", capacity=10.0, start_location_id="depot", end_location_id="depot")
        ]
        self.deliveries = []

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_with_no_deliveries(self, mock_solve, mock_create_matrix):
        """Should handle when there are no deliveries."""
        mock_create_matrix.return_value = (np.array([[0.0]]), ["depot"])
        mock_solve.return_value = {
            'status': 'success',
            'routes': [[0]],
            'total_distance': 0.0,
            'assigned_vehicles': {'vehicle1': 0},
            'unassigned_deliveries': []
        }
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=[]
        )
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['total_distance'], 0.0)
        self.assertEqual(result['routes'][0], [0])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_invalid_depot_index(self, mock_solve, mock_create_matrix):
        """Should fall back to index 0 when no depot is marked."""
        locations = [Location(id="node0", name="Node 0", latitude=0.0, longitude=0.0)]
        mock_create_matrix.return_value = (np.array([[0.0]]), ["node0"])
        mock_solve.return_value = {
            'status': 'success',
            'routes': [[0]],
            'total_distance': 0.0,
            'assigned_vehicles': {'vehicle1': 0},
            'unassigned_deliveries': []
        }
        result = self.service.optimize_routes(
            locations=locations,
            vehicles=self.vehicles,
            deliveries=[]
        )
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['routes'][0], [0])

    @patch('route_optimizer.core.distance_matrix.DistanceMatrixBuilder.create_distance_matrix')
    @patch('route_optimizer.core.ortools_optimizer.ORToolsVRPSolver.solve')
    def test_optimize_failure_case(self, mock_solve, mock_create_matrix):
        """Should return failure and skip enrichment if solver fails."""
        mock_create_matrix.return_value = (np.array([[0.0]]), ["depot"])
        mock_solve.return_value = {'status': 'failure'}
        result = self.service.optimize_routes(
            locations=self.locations,
            vehicles=self.vehicles,
            deliveries=self.deliveries
        )
        self.assertEqual(result['status'], 'failure')
        self.assertNotIn('detailed_routes', result)

