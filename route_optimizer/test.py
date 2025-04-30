import pytest
import requests
from optimizer import RouteOptimizer
from app import app
import math

# Pytest fixtures for reusable test data
@pytest.fixture
def simple_locations():
    """Simple test locations with depot and two points"""
    return [
        {'id': 'depot', 'coordinates': (0, 0), 'load': 0},
        {'id': 'A', 'coordinates': (0.018, 0.018), 'load': 1},
        {'id': 'B', 'coordinates': (0.027, 0.027), 'load': 2}
    ]

@pytest.fixture
def sri_lanka_locations():
    """Real-world Sri Lanka locations"""
    return [
        {'id': 'Warehouse_Colombo', 'coordinates': [6.9271, 79.8612], 'load': 0},
        {'id': 'Restaurant_Kandy', 'coordinates': [7.2906, 80.6337], 'load': 3},
        {'id': 'Cafe_Galle', 'coordinates': [6.0535, 80.2210], 'load': 2},
        {'id': 'Shop_Negombo', 'coordinates': [7.2088, 79.8380], 'load': 1},
        {'id': 'Hotel_NuwaraEliya', 'coordinates': [6.9497, 80.7891], 'load': 4},
        {'id': 'Market_Anuradhapura', 'coordinates': [8.3114, 80.4037], 'load': 2},
        {'id': 'Office_Jaffna', 'coordinates': [9.6615, 80.0255], 'load': 3}
    ]

@pytest.fixture
def vehicle_capacities():
    """Standard vehicle capacities"""
    return [5, 6, 4]

@pytest.fixture
def flask_client():
    """Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# RouteOptimizer tests
def test_basic_optimization(simple_locations, vehicle_capacities):
    """Test basic route optimization with simple locations"""
    optimizer = RouteOptimizer(simple_locations, [3])
    result = optimizer.solve()
    
    assert result is not None, "No solution found"
    assert len(result) == 1, "Expected one route"
    assert result[0]['route'][0] == 'depot', "Route must start at depot"
    assert result[0]['route'][-1] == 'depot', "Route must end at depot"
    assert set(result[0]['route'][1:-1]) == {'A', 'B'}, "Route must visit A and B"
    assert result[0]['load'] == 3, f"Expected load 3, got {result[0]['load']}"
    assert result[0]['distance'] > 0, "Distance must be non-zero"

def test_sri_lanka_optimization(sri_lanka_locations, vehicle_capacities):
    """Test optimization with Sri Lanka locations"""
    optimizer = RouteOptimizer(sri_lanka_locations, vehicle_capacities)
    result = optimizer.solve()
    
    assert result is not None, "No solution found"
    assert len(result) <= len(vehicle_capacities), "Too many routes"
    
    # Check all non-depot locations are visited exactly once
    all_locations = set(loc['id'] for loc in sri_lanka_locations[1:])
    visited_locations = set()
    for route in result:
        assert route['route'][0] == 'Warehouse_Colombo', f"Route {route['vehicle_id']} must start at depot"
        assert route['route'][-1] == 'Warehouse_Colombo', f"Route {route['vehicle_id']} must end at depot"
        assert route['load'] <= vehicle_capacities[route['vehicle_id']], f"Route {route['vehicle_id']} exceeds capacity"
        assert route['distance'] >= 10000, f"Route {route['vehicle_id']} distance {route['distance']} too small"
        visited_locations.update(route['route'][1:-1])
    
    assert visited_locations == all_locations, f"Missing locations: {all_locations - visited_locations}"

def test_invalid_depot_load(simple_locations):
    """Test depot with non-zero load"""
    bad_locations = simple_locations.copy()
    bad_locations[0]['load'] = 1
    with pytest.raises(ValueError, match="Depot must have 0 load"):
        RouteOptimizer(bad_locations, [3]).solve()

def test_empty_locations():
    """Test empty locations list"""
    with pytest.raises(ValueError, match="No delivery locations provided"):
        RouteOptimizer([], [3]).solve()

def test_no_vehicles(simple_locations):
    """Test no vehicles provided"""
    with pytest.raises(ValueError, match="At least one vehicle required"):
        RouteOptimizer(simple_locations, []).solve()

def test_haversine_distance():
    """Test Haversine distance calculation"""
    optimizer = RouteOptimizer([], [3])
    distance = optimizer.haversine_distance(6.9271, 79.8612, 7.2906, 80.6337)  # Colombo to Kandy
    assert 93000 < distance < 95000, f"Expected ~94.3km, got {distance}m"

def test_distance_matrix(simple_locations):
    """Test distance matrix creation"""
    optimizer = RouteOptimizer(simple_locations, [3])
    matrix = optimizer.create_distance_matrix()
    assert len(matrix) == 3, "Matrix should have 3 rows"
    assert len(matrix[0]) == 3, "Matrix should have 3 columns"
    assert matrix[0][0] == 0, "Depot to depot distance should be 0"
    assert matrix[1][2] == matrix[2][1], "Matrix should be symmetric"
    assert matrix[1][2] > 0, "Distance between A and B should be non-zero"

def test_no_solution(simple_locations):
    """Test scenario with no feasible solution (insufficient capacity)"""
    optimizer = RouteOptimizer(simple_locations, [1])  # Capacity too low
    result = optimizer.solve()
    assert result is None, "Expected no solution due to insufficient capacity"

# Flask API tests
def test_api_success_simple(flask_client, simple_locations):
    """Test API with simple valid input"""
    response = flask_client.post(
        '/optimize',
        json={
            'locations': simple_locations,
            'vehicle_capacities': [3]
        }
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert 'routes' in data, "Response must include routes"
    assert len(data['routes']) == 1, "Expected one route"
    assert 'coordinates' in data['routes'][0], "Route must include coordinates"
    assert data['routes'][0]['distance'] > 0, "Distance must be non-zero"
    assert data['routes'][0]['load'] == 3, "Expected load 3"
    assert len(data['routes'][0]['coordinates']) == len(data['routes'][0]['route']), "Coordinates must match route length"

def test_api_success_sri_lanka(flask_client, sri_lanka_locations, vehicle_capacities):
    """Test API with Sri Lanka locations"""
    response = flask_client.post(
        '/optimize',
        json={
            'locations': sri_lanka_locations,
            'vehicle_capacities': vehicle_capacities
        }
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert 'routes' in data, "Response must include routes"
    
    all_locations = set(loc['id'] for loc in sri_lanka_locations[1:])
    visited_locations = set()
    for route in data['routes']:
        assert 'coordinates' in route, f"Route {route['vehicle_id']} missing coordinates"
        assert route['distance'] >= 10000, f"Route {route['vehicle_id']} distance {route['distance']} too small"
        assert len(route['coordinates']) == len(route['route']), "Coordinates must match route length"
        visited_locations.update(route['route'][1:-1])
    
    assert visited_locations == all_locations, f"Missing locations: {all_locations - visited_locations}"

def test_api_missing_locations(flask_client):
    """Test API with missing locations"""
    response = flask_client.post(
        '/optimize',
        json={'vehicle_capacities': [3]}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert 'error' in data, "Response must include error"
    assert 'Missing required field' in data['error'], "Expected missing field error"

def test_api_missing_capacities(flask_client, simple_locations):
    """Test API with missing vehicle capacities"""
    response = flask_client.post(
        '/optimize',
        json={'locations': simple_locations}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert 'error' in data, "Response must include error"
    assert 'Missing required field' in data['error'], "Expected missing field error"

def test_api_invalid_coordinates(flask_client):
    """Test API with invalid coordinates"""
    bad_locations = [
        {'id': 'depot', 'coordinates': [0, 0], 'load': 0},
        {'id': 'A', 'coordinates': 'invalid', 'load': 1}
    ]
    response = flask_client.post(
        '/optimize',
        json={'locations': bad_locations, 'vehicle_capacities': [3]}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert 'error' in data, "Response must include error"

def test_api_no_solution(flask_client, simple_locations):
    """Test API with no feasible solution"""
    response = flask_client.post(
        '/optimize',
        json={'locations': simple_locations, 'vehicle_capacities': [1]}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert 'error' in data, "Response must include error"
    assert 'No solution found' in data['error'], "Expected no solution error"

def test_api_internal_error(flask_client, simple_locations, monkeypatch):
    """Test API with internal server error (mocked failure)"""
    def mock_solve(*args, **kwargs):
        raise Exception("Mocked internal error")
    
    monkeypatch.setattr(RouteOptimizer, 'solve', mock_solve)
    response = flask_client.post(
        '/optimize',
        json={'locations': simple_locations, 'vehicle_capacities': [3]}
    )
    assert response.status_code == 500, f"Expected 500, got {response.status_code}"
    data = response.get_json()
    assert 'error' in data, "Response must include error"
    assert 'Internal server error' in data['error'], "Expected internal error"

if __name__ == "__main__":
    pytest.main(["-v", __file__])