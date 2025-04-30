from flask import Flask, request, jsonify
from flask_cors import CORS
from optimizer import RouteOptimizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/optimize', methods=['POST'])
def optimize_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        if 'locations' not in data:
            return jsonify({'error': 'Missing required field: locations'}), 400
        if 'vehicle_capacities' not in data:
            return jsonify({'error': 'Missing required field: vehicle_capacities'}), 400

        locations = data['locations']
        vehicle_capacities = data['vehicle_capacities']

        # Validate input formats
        if not isinstance(locations, list) or not locations:
            return jsonify({'error': 'Locations must be a non-empty list'}), 400
        if not isinstance(vehicle_capacities, list) or not vehicle_capacities:
            return jsonify({'error': 'Vehicle capacities must be a non-empty list'}), 400

        # Validate location fields
        for loc in locations:
            if not all(key in loc for key in ['id', 'coordinates', 'load']):
                return jsonify({'error': f'Missing required fields in location: {loc.get("id", "unknown")}'}), 400
            if not isinstance(loc['coordinates'], (list, tuple)) or len(loc['coordinates']) != 2:
                return jsonify({'error': f'Invalid coordinates for location: {loc.get("id", "unknown")}'}), 400
            if not all(isinstance(coord, (int, float)) for coord in loc['coordinates']):
                return jsonify({'error': f'Coordinates must be numbers for location: {loc.get("id", "unknown")}'}), 400
            if not isinstance(loc['load'], (int, float)):
                return jsonify({'error': f'Load must be a number for location: {loc.get("id", "unknown")}'}), 400

        # Validate vehicle capacities
        if not all(isinstance(cap, (int, float)) and cap > 0 for cap in vehicle_capacities):
            return jsonify({'error': 'Vehicle capacities must be positive numbers'}), 400

        optimizer = RouteOptimizer(locations, vehicle_capacities)
        result = optimizer.solve()

        if not result:
            return jsonify({'error': 'No solution found'}), 400

        # Add coordinates to each route
        location_map = {loc['id']: loc['coordinates'] for loc in locations}
        for route in result:
            try:
                route['coordinates'] = [location_map[loc_id] for loc_id in route['route']]
            except KeyError as e:
                return jsonify({'error': f'Missing location ID: {e}'}), 400

        return jsonify({'routes': result}), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)