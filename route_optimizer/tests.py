from django.test import TestCase

# Create your tests here.
from route_optimizer.optimizer import RouteOptimizer

class RouteOptimizerTest(TestCase):
    def setUp(self):
        """Set up test data for RouteOptimizer tests."""
        self.simple_locations = [
            {'id': 'depot', 'coordinates': [0, 0], 'load': 0},
            {'id': 'A', 'coordinates': [0.018, 0.018], 'load': 1},
            {'id': 'B', 'coordinates': [0.027, 0.027], 'load': 2}
        ]
        self.vehicle_capacities = [3]
        self.sri_lanka_locations = [
            {'id': 'Warehouse_Colombo', 'coordinates': [6.9271, 79.8612], 'load': 0},
            {'id': 'Restaurant_Kandy', 'coordinates': [7.2906, 80.6337], 'load': 3},
            {'id': 'Cafe_Galle', 'coordinates': [6.0535, 80.2210], 'load': 2}
        ]

    def test_successful_optimization_simple(self):
        """Test route optimization with simple locations."""
        optimizer = RouteOptimizer(self.simple_locations, self.vehicle_capacities)
        result = optimizer.solve()
        
        self.assertIsNotNone(result, "Expected a valid solution")
        self.assertEqual(len(result), 1, "Expected one route")
        self.assertEqual(result[0]['route'][0], 'depot', "Route must start at depot")
        self.assertEqual(result[0]['route'][-1], 'depot', "Route must end at depot")
        self.assertEqual(set(result[0]['route'][1:-1]), {'A', 'B'}, "Route must visit A and B")
        self.assertEqual(result[0]['load'], 3, "Expected total load of 3")
        self.assertGreater(result[0]['distance'], 0, "Distance must be non-zero")

    def test_successful_optimization_sri_lanka(self):
        """Test route optimization with Sri Lanka locations."""
        optimizer = RouteOptimizer(self.sri_lanka_locations, [5])
        result = optimizer.solve()
        
        self.assertIsNotNone(result, "Expected a valid solution")
        self.assertEqual(len(result), 1, "Expected one route")
        self.assertEqual(result[0]['route'][0], 'Warehouse_Colombo', "Route must start at depot")
        self.assertEqual(result[0]['route'][-1], 'Warehouse_Colombo', "Route must end at depot")
        self.assertEqual(set(result[0]['route'][1:-1]), {'Restaurant_Kandy', 'Cafe_Galle'}, 
                        "Route must visit Kandy and Galle")
        self.assertEqual(result[0]['load'], 5, "Expected total load of 5")
        self.assertGreaterEqual(result[0]['distance'], 10000, "Distance must be significant")

    def test_insufficient_vehicle_capacity(self):
        """Test optimization with insufficient vehicle capacity."""
        optimizer = RouteOptimizer(self.simple_locations, [1])  # Capacity too low
        result = optimizer.solve()
        
        self.assertIsNone(result, "Expected no solution due to insufficient capacity")

    def test_invalid_depot_load(self):
        """Test initialization with non-zero depot load."""
        invalid_locations = self.simple_locations.copy()
        invalid_locations[0]['load'] = 1
        optimizer = RouteOptimizer(invalid_locations, self.vehicle_capacities)
        
        with self.assertRaises(ValueError) as context:
            optimizer.solve()
        self.assertEqual(str(context.exception), "Depot must have 0 load")

    def test_empty_locations(self):
        """Test initialization with empty locations list."""
        optimizer = RouteOptimizer([], self.vehicle_capacities)
        
        with self.assertRaises(ValueError) as context:
            optimizer.solve()
        self.assertEqual(str(context.exception), "No delivery locations provided")

    def test_no_vehicles(self):
        """Test initialization with no vehicles."""
        optimizer = RouteOptimizer(self.simple_locations, [])
        
        with self.assertRaises(ValueError) as context:
            optimizer.solve()
        self.assertEqual(str(context.exception), "At least one vehicle required")

    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        optimizer = RouteOptimizer(self.sri_lanka_locations, self.vehicle_capacities)
        distance = optimizer.haversine_distance(6.9271, 79.8612, 7.2906, 80.6337)  # Colombo to Kandy
        
        self.assertGreater(distance, 93000, "Distance should be more than 93km")
        self.assertLess(distance, 95000, "Distance should be less than 95km")

    def test_distance_matrix(self):
        """Test distance matrix creation."""
        optimizer = RouteOptimizer(self.simple_locations, self.vehicle_capacities)
        matrix = optimizer.create_distance_matrix()
        
        self.assertEqual(len(matrix), 3, "Matrix should have 3 rows")
        self.assertEqual(len(matrix[0]), 3, "Matrix should have 3 columns")
        self.assertEqual(matrix[0][0], 0, "Depot to depot distance should be 0")
        self.assertEqual(matrix[1][2], matrix[2][1], "Matrix should be symmetric")
        self.assertGreater(matrix[1][2], 0, "Distance between A and B should be non-zero")