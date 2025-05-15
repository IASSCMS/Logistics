# Logistics

- route_optimizer/ (Standalone service)
  - Inputs: delivery locations, vehicle capacities
  - Output: optimized delivery route
  - Purpose: The brain of the logistics module
  - Develop this first so you can test assignment logic early
 
- map_service/ (Optional)
  - Utility/service to fetch real-world distances
  - Can be a local function or API-based (OpenRouteService / OpenStreetMap)
  - Can be skipped initially, use dummy distance matrix
 
- fleet/ (Django app)
  - Models: Vehicle, Status, Capacity, Location
  - REST APIs to get available vehicles, update location/status
  - You’ll need this to match vehicles with optimized route
 
- assignment/ (Django app)
  - Inputs: optimized route + available vehicles (from fleet)
  - Logic: assign deliveries to vehicle
  - Output: assignment events, persisted records
  - Triggers optimizer and manages dispatching
 
- scheduler/ (Lambda or Celery)
  - Automatically triggers assignment + route_optimizer daily/hourly
  - Optional for early dev, but crucial for automation
 
- monitoring/ (Django app)
  - Captures logs, alerts, failed deliveries, delays
  - Optional dashboard with charts and status
  - Could connect with Kafka or DB log events from assignment
- shipments/ (Django app) 
  - Models: Shipment (order_id, origin, destination, status)
  - Status Lifecycle: pending → scheduled → dispatched → in_transit → delivered/failed
  - APIs: Create shipment, update status, track delivery progress
  - Decoupled from Warehouse via primitive IDs (warehouse_id)
  - Triggered by Order events (async/REST), manages physical movement of goods

---
# Getting Started
### 1. ✅ Clone the Repository

```bash
git clone https://github.com/IASSCMS/Logistics.git
cd Logistics
```

---

### 2. 🐍 Create & Activate Virtual Environment

#### On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. 📦 Install Dependencies

Make sure your virtual environment is activated, then run:

```bash
pip install -r requirements.txt
```

---

### 4. ⚙️ Apply Migrations

```bash
python manage.py migrate
```

---

### 5. 🚦 Run the Development Server

```bash
python manage.py runserver
```

---

### 6. 📚 View API Documentation (Swagger)

Once the server is running, open your browser and go to:

```
http://127.0.0.1:8000/swagger/
```

You’ll see an interactive **Swagger UI** listing all available API endpoints (e.g., `/api/fleet/vehicles/`).

```
Logistics
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        ├─ lastfailed
│        ├─ nodeids
│        └─ stepwise
├─ assignment
│  ├─ admin.py
│  ├─ apps.py
│  ├─ migrations
│  │  ├─ 0001_initial.py
│  │  └─ __init__.py
│  ├─ models.py
│  ├─ serializers.py
│  ├─ tests.py
│  ├─ urls.py
│  ├─ views.py
│  └─ __init__.py
├─ docker-compose.yml
├─ fleet
│  ├─ admin.py
│  ├─ apps.py
│  ├─ migrations
│  │  ├─ 0001_initial.py
│  │  ├─ 0002_vehicle_created_at_vehicle_current_latitude_and_more.py
│  │  ├─ 0003_remove_fuelrecord_vehicle_and_more.py
│  │  └─ __init__.py
│  ├─ models
│  │  ├─ core.py
│  │  ├─ extended_models.py
│  │  └─ __init__.py
│  ├─ serializers
│  │  ├─ fuel.py
│  │  ├─ maintenance.py
│  │  ├─ trip.py
│  │  ├─ vehicle.py
│  │  └─ __init__.py
│  ├─ tests
│  │  ├─ test_fuel.py
│  │  ├─ test_fuel_api.py
│  │  ├─ test_maintenance.py
│  │  ├─ test_maintenance_api.py
│  │  ├─ test_trip.py
│  │  ├─ test_trip_api.py
│  │  ├─ test_vehicle.py
│  │  ├─ test_vehicle_api.py
│  │  └─ __init__.py
│  ├─ urls.py
│  ├─ views
│  │  ├─ fuel.py
│  │  ├─ maintenance.py
│  │  ├─ trip.py
│  │  ├─ vehicle.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ LICENSE
├─ logistics_core
│  ├─ asgi.py
│  ├─ settings.py
│  ├─ urls.py
│  ├─ wsgi.py
│  └─ __init__.py
├─ manage.py
├─ monitoring
│  ├─ admin.py
│  ├─ apps.py
│  ├─ migrations
│  │  └─ __init__.py
│  ├─ models.py
│  ├─ tests.py
│  ├─ views.py
│  └─ __init__.py
├─ order_simulator.py
├─ README.md
├─ requirements.txt
├─ route_optimizer
│  ├─ admin.py
│  ├─ api
│  │  ├─ serializers.py
│  │  ├─ urls.py
│  │  ├─ views.py
│  │  └─ __init__.py
│  ├─ apps.py
│  ├─ core
│  │  ├─ constants.py
│  │  ├─ dijkstra.py
│  │  ├─ distance_matrix.py
│  │  ├─ ortools_optimizer.py
│  │  ├─ types_1.py
│  │  └─ __init__.py
│  ├─ migrations
│  │  ├─ 0001_initial.py
│  │  └─ __init__.py
│  ├─ models.py
│  ├─ README.md
│  ├─ services
│  │  ├─ depot_service.py
│  │  ├─ external_data_service.py
│  │  ├─ optimization_service.py
│  │  ├─ path_annotation_service.py
│  │  ├─ rerouting_service.py
│  │  ├─ route_stats_service.py
│  │  ├─ traffic_service.py
│  │  └─ __init__.py
│  ├─ settings.py
│  ├─ tests
│  │  ├─ api
│  │  │  ├─ test_serializers.py
│  │  │  └─ test_views.py
│  │  ├─ conftest.py
│  │  ├─ core
│  │  │  ├─ test_dijkstra.py
│  │  │  ├─ test_distance_matrix.py
│  │  │  ├─ test_ortools_optimizer.py
│  │  │  ├─ test_types.py
│  │  │  └─ __init__.py
│  │  ├─ services
│  │  │  ├─ test_depot_service.py
│  │  │  ├─ test_external_data_service.py
│  │  │  ├─ test_optimization_service.py
│  │  │  ├─ test_path_annotation_service.py
│  │  │  ├─ test_rerouting_service.py
│  │  │  ├─ test_route_stats_service.py
│  │  │  ├─ test_traffic_service.py
│  │  │  └─ __init__.py
│  │  ├─ test_models.py
│  │  ├─ test_settings.py
│  │  ├─ utils
│  │  │  ├─ test_env_loader.py
│  │  │  └─ test_helpers.py
│  │  └─ __init__.py
│  ├─ utils
│  │  ├─ env_loader.py
│  │  ├─ helpers.py
│  │  └─ __init__.py
│  ├─ views.py
│  └─ __init__.py
└─ shipments
   ├─ admin.py
   ├─ apps.py
   ├─ consumers
   │  └─ order_events.py
   ├─ management
   │  └─ commands
   │     └─ consume_orders.py
   ├─ migrations
   │  ├─ 0001_initial.py
   │  └─ __init__.py
   ├─ models.py
   ├─ serializers.py
   ├─ tests
   │  ├─ test_api.py
   │  ├─ test_consumer.py
   │  ├─ test_integration_kafka.py
   │  └─ __init__.py
   ├─ urls.py
   ├─ views.py
   └─ __init__.py

```