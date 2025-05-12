# Route Optimizer Module

## Overview

The **Route Optimizer** module is a comprehensive Django app designed to calculate optimal routes for a fleet of vehicles to make deliveries (or pickups) to a set of locations. It considers various constraints such as vehicle capacities, time windows, traffic conditions, and operational costs. The module can use either local calculations (like Haversine for distance) or external APIs (like Google Maps Distance Matrix API) for more accurate real-world data. It also supports dynamic rerouting in response to real-time events.

---

## Core Files (`route_optimizer/core/`)

The `core` directory contains the fundamental algorithms, data type definitions, and constants that form the backbone of the route optimization logic.

### 1. `constants.py` ([Logistics\route_optimizer\core\constants.py](file:///Logistics\route_optimizer\core\constants.py))

*   **Functionality**:
    *   Defines various constants used throughout the optimization process.
    *   Includes scaling factors (e.g., `DISTANCE_SCALING_FACTOR`, `CAPACITY_SCALING_FACTOR`, `TIME_SCALING_FACTOR`) required by OR-Tools to work with integer arithmetic.
    *   Specifies safety bounds for distance and time values (e.g., `MAX_SAFE_DISTANCE`, `MAX_SAFE_TIME`).
*   **Important Points**:
    *   **Consistency is Key**: Ensure these constants are consistently used and understood across all modules, especially between the main code and tests. Mismatches in values like `MAX_SAFE_DISTANCE` have caused issues previously.
    *   **Scaling Factor Impact**: The scaling factors directly affect the precision and behavior of the OR-Tools solver. Adjustments might be needed based on the typical range of input values.
    *   The commented-out section suggests different scaling strategies might have been considered; the current active set is what's in use.

### 2. `dijkstra.py` ([Logistics\route_optimizer\core\dijkstra.py](file:///Logistics\route_optimizer\core\dijkstra.py))

*   **Functionality**:
    *   Provides an implementation of Dijkstra's algorithm.
    *   `DijkstraPathFinder` class offers:
        *   `calculate_shortest_path(graph, start, end)`: Finds the shortest path between a single pair of nodes.
        *   `calculate_all_shortest_paths(graph, nodes)`: Calculates shortest paths between all specified pairs of nodes.
        *   `_validate_non_negative_weights(graph)`: Ensures no negative edge weights.
*   **Important Points**:
    *   **Negative Weights**: Raises a `ValueError` for negative weights. If negative weights are needed, an alternative like Bellman-Ford would be required.
    *   **Graph Representation**: Expects graph as a dictionary of dictionaries (adjacency list with weights).
    *   **Use Case**: Used by `PathAnnotator` for detailed path segments when not using external APIs.

### 3. `distance_matrix.py` ([Logistics\route_optimizer\core\distance_matrix.py](file:///Logistics\route_optimizer\core\distance_matrix.py))

*   **Functionality**:
    *   `DistanceMatrixBuilder` class for creating and managing distance matrices.
    *   Supports Haversine, Euclidean, and Google Maps Distance Matrix API calculations.
    *   Includes caching (`DistanceMatrixCache` model), API request retries, address formatting, response processing, matrix sanitization (`_sanitize_distance_matrix`), traffic factor application (`add_traffic_factors`, `_apply_traffic_safely`), and matrix-to-graph conversion.
*   **Important Points**:
    *   **API Key**: Relies on `GOOGLE_MAPS_API_KEY` from `settings.py`.
    *   **API Quotas**: Caching is vital to manage API usage.
    *   **Fallback Behavior**: Falls back to Haversine if API fails.
    *   **Sanitization**: `_sanitize_distance_matrix` is crucial for clean numerical data, replacing `NaN`, `inf`, etc., with `MAX_SAFE_DISTANCE` or 0.
    *   **Traffic Application**: `_apply_traffic_safely` includes bounds checking for traffic factors.

### 4. `ortools_optimizer.py` ([Logistics\route_optimizer\core\ortools_optimizer.py](file:///Logistics\route_optimizer\core\ortools_optimizer.py))

*   **Functionality**:
    *   `ORToolsVRPSolver` class for solving Vehicle Routing Problems (VRP) using Google OR-Tools.
    *   `solve(...)`: Basic VRP with capacity constraints.
    *   `solve_with_time_windows(...)`: VRP with time window constraints.
    *   Handles vehicle start/end locations, capacities, and cost minimization.
*   **Important Points**:
    *   **Integer Scaling**: Critical for OR-Tools; uses scaling factors from `constants.py`.
    *   **Depot Handling**: `depot_index` is fundamental.
    *   **Callbacks**: Distance, demand, and time callbacks are essential.
    *   **Solution Interpretation**: Output is parsed into `OptimizationResult`.
    *   **Time Limits**: Configurable `time_limit_seconds`.
    *   **Empty Problem**: Creates depot-to-depot routes if no deliveries.

### 5. `types_1.py` ([Logistics\route_optimizer\core\types_1.py](file:///Logistics\route_optimizer\core\types_1.py))

*   **Functionality**:
    *   Defines core Data Transfer Objects (DTOs) using `dataclass`:
        *   `Location`: Geographic point with coordinates, depot status, time windows, service time.
        *   `OptimizationResult`: Standardized output format.
        *   `RouteSegment`: Details of a path segment.
        *   `DetailedRoute`: Comprehensive vehicle route description.
        *   `ReroutingInfo`: Information for rerouting operations.
    *   `validate_optimization_result(result)`: Validates `OptimizationResult` structure.
*   **Important Points**:
    *   **Standardization**: Ensures consistent data handling.
    *   **Validation**: `validate_optimization_result` is key for data integrity.
    *   **Mutability**: Dataclasses are mutable by default.

---

## Services Files (`route_optimizer/services/`)

The `services` directory orchestrates core logic and handles higher-level tasks.

### 1. `depot_service.py` ([Logistics\route_optimizer\services\depot_service.py](file:///Logistics\route_optimizer\services\depot_service.py))

*   **Functionality**:
    *   `DepotService` class for depot location utilities.
    *   `get_nearest_depot(locations)`: Identifies a depot. Defaults to the first depot found or the first location.
    *   `find_depot_index(locations)`: Returns index of the depot. Defaults to 0.
*   **Important Points**:
    *   **Depot Assumption**: Simple logic for multiple depots (returns first). Fallback to the first location if no explicit depot.

### 2. `external_data_service.py` ([Logistics\route_optimizer\services\external_data_service.py](file:///Logistics\route_optimizer\services\external_data_service.py))

*   **Functionality**:
    *   `ExternalDataService` for fetching external data (traffic, weather, roadblocks).
    *   Currently provides mock data if `use_mocks` is true or real APIs are unimplemented.
    *   Includes helpers for mock data generation and combining factors.
*   **Important Points**:
    *   **Mock Data**: Real API integrations needed for production.
    *   **API Keys**: Would require key management if real APIs were used.

### 3. `optimization_service.py` ([Logistics\route_optimizer\services\optimization_service.py](file:///Logistics\route_optimizer\services\optimization_service.py))

*   **Functionality**:
    *   `OptimizationService`: Main orchestrator for route optimization.
    *   `optimize_routes(...)`: Primary method. Validates inputs, creates/sanitizes distance matrix, applies traffic, determines depot, calls VRP solver, converts/enriches result with detailed paths and summary statistics.
    *   Helper methods for input validation, path/stats addition, result conversion, matrix sanitization.
*   **Important Points**:
    *   **Central Orchestrator**: Ties many components together.
    *   **Error Handling**: General `try-except` and specific input validation.
    *   **API Usage Control**: `use_api` flag and `USE_API_BY_DEFAULT` setting.
    *   **Result Enrichment**: Multi-step process for detailed results.
    *   **Backward Compatibility**: `_add_detailed_paths` handles `dict` and `OptimizationResult`.

### 4. `path_annotation_service.py` ([Logistics\route_optimizer\services\path_annotation_service.py](file:///Logistics\route_optimizer\services\path_annotation_service.py))

*   **Functionality**:
    *   `PathAnnotator` class for adding detailed segment-by-segment path information.
    *   `annotate(result, graph_or_matrix)`: Uses a `path_finder` (e.g., `DijkstraPathFinder`) for segment details.
    *   Accepts graph or distance matrix. Handles `dict` and `OptimizationResult`.
    *   `_add_summary_statistics` helper ensures `detailed_routes` structure.
*   **Important Points**:
    *   **Dependency**: Relies on an injected `path_finder`.
    *   **Error Handling**: Logs errors and adds placeholders for failed path calculations.

### 5. `rerouting_service.py` ([Logistics\route_optimizer\services\rerouting_service.py](file:///Logistics\route_optimizer\services\rerouting_service.py))

*   **Functionality**:
    *   `ReroutingService` for dynamic route adjustments.
    *   Methods: `reroute_for_traffic`, `reroute_for_delay`, `reroute_for_roadblock`.
    *   Helpers: `_get_remaining_deliveries`, `_update_vehicle_positions`.
    *   Relies on `OptimizationService` for re-optimization.
*   **Important Points**:
    *   **State Management**: Accurate current state (completed deliveries, vehicle positions) is crucial; current helpers are placeholders.
    *   **Complexity**: Rerouting triggers a new, potentially intensive optimization.

### 6. `route_stats_service.py` ([Logistics\route_optimizer\services\route_stats_service.py](file:///Logistics\route_optimizer\services\route_stats_service.py))

*   **Functionality**:
    *   `RouteStatsService` calculates and adds statistics to the optimization result.
    *   `add_statistics(result, vehicles)`: Calculates vehicle/total costs, aggregates total stops/distance/vehicles used. Handles `OptimizationResult` and `dict`.
*   **Important Points**:
    *   **Cost Calculation**: Uses `fixed_cost` and `cost_per_km` from `Vehicle` objects.
    *   **Data Dependency**: Needs `detailed_routes` with segment distances for accurate costs.

### 7. `traffic_service.py` ([Logistics\route_optimizer\services\traffic_service.py](file:///Logistics\route_optimizer\services\traffic_service.py))

*   **Functionality**:
    *   `TrafficService` for traffic-related information.
    *   `apply_traffic_factors(...)`: Wraps `DistanceMatrixBuilder.add_traffic_factors`.
    *   `create_road_graph(locations)`: Creates a road network graph. Potential integration point for Google Maps API for accurate topology if API key is used. Currently basic.
*   **Important Points**:
    *   **API Integration**: `create_road_graph` is key for potential Google Maps API use for detailed pathing when `OptimizationService`'s `use_api_flag` is true.

### 8. `vrp_solver.py` ([Logistics\route_optimizer\services\vrp_solver.py](file:///Logistics\route_optimizer\services\vrp_solver.py))

*   **Functionality**:
    *   Contains a standalone `solve_with_time_windows(...)` function, similar to the method in `core/ortools_optimizer.py`.
*   **Important Points**:
    *   **Redundancy/Placement**: May need refactoring. `core/ortools_optimizer.py` should be the primary OR-Tools solver implementation. This might be legacy or a specialized helper.

---

## Settings File (`route_optimizer/settings.py`)

*   [Logistics\route_optimizer\settings.py](file:///Logistics\route_optimizer\settings.py)
*   **Functionality**:
    *   Manages app configurations. Loads environment variables from `env_var.env`.
    *   Defines `GOOGLE_MAPS_API_KEY`, `GOOGLE_MAPS_API_URL`, `USE_API_BY_DEFAULT`.
    *   API request settings: `MAX_RETRIES`, `BACKOFF_FACTOR`, `RETRY_DELAY_SECONDS`.
    *   `CACHE_EXPIRY_DAYS`.
    *   `TESTING` flag.
*   **Important Points**:
    *   **Environment Variables**: Critical for API keys and sensitive data. `env_loader` assists local setup.
    *   **API Key Security**: `GOOGLE_MAPS_API_KEY` is vital; `.env` file must be in `.gitignore`.
    *   **Test Mode**: Allows different configurations for testing.

---

## Utils Files (`route_optimizer/utils/`)

### 1. `env_loader.py` ([Logistics\route_optimizer\utils\env_loader.py](file:///Logistics\route_optimizer\utils\env_loader.py))

*   **Functionality**:
    *   `load_env_from_file(file_path)`: Loads `KEY=VALUE` pairs from a file into `os.environ`.
*   **Important Points**:
    *   **Local Development**: Useful for simulating production env vars locally.
    *   **Security**: The env file itself should be secure and not version-controlled.

### 2. `helpers.py` ([Logistics\route_optimizer\utils\helpers.py](file:///Logistics\route_optimizer\utils\helpers.py))

*   **Functionality**:
    *   Collection of miscellaneous utility functions: time conversions, Haversine calculation, route formatting, basic stats calculation, distance/time matrix creation, isolated node detection, safe JSON dumps, duration formatting.
*   **Important Points**:
    *   **Redundancy**: Some functions might overlap with more specialized classes (e.g., Haversine vs. `DistanceMatrixBuilder`). Potential for refactoring.
    *   **Generic Utilities**: Small, focused helpers for general use.

---

## Other Project Files

*   **`admin.py`** ([Logistics\route_optimizer\admin.py](file:///Logistics\route_optimizer\admin.py)): For Django admin interface. Currently empty.
*   **`api/` directory**:
    *   **`serializers.py`** ([Logistics\route_optimizer\api\serializers.py](file:///Logistics\route_optimizer\api\serializers.py)): DRF serializers for API data validation and conversion.
    *   **`urls.py`** ([Logistics\route_optimizer\api\urls.py](file:///Logistics\route_optimizer\api\urls.py)): API URL patterns.
    *   **`views.py`** ([Logistics\route_optimizer\api\views.py](file:///Logistics\route_optimizer\api\views.py)): API views (`OptimizeRoutesView`, `RerouteView`) handling HTTP requests and responses.
*   **`apps.py`** ([Logistics\route_optimizer\apps.py](file:///Logistics\route_optimizer\apps.py)): Django app configuration.
*   **`migrations/`**: Django database migration files (e.g., for `DistanceMatrixCache`).
*   **`models.py`** ([Logistics\route_optimizer\models.py](file:///Logistics\route_optimizer\models.py)): Dataclasses for `Vehicle` and `Delivery`; Django model for `DistanceMatrixCache`.
*   **`tests/`**: Unit and integration tests.
    *   **`conftest.py`** ([Logistics\route_optimizer\tests\conftest.py](file:///Logistics\route_optimizer\tests\conftest.py)): Pytest configuration.
    *   **`test_settings.py`** ([Logistics\route_optimizer\tests\test_settings.py](file:///Logistics\route_optimizer\tests\test_settings.py)): Django settings for tests.
*   **`views.py` (root)** ([Logistics\route_optimizer\views.py](file:///Logistics\route_optimizer\views.py)): Standard Django views file, currently empty.

---

This comprehensive overview should help in understanding the structure, functionality, and key considerations of the `route_optimizer` module.