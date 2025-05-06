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

---

```
Logistics
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
├─ fleet
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
├─ README.md
├─ requirements.txt
└─ route_optimizer
   ├─ admin.py
   ├─ api
   │  ├─ serializers.py
   │  ├─ urls.py
   │  ├─ views.py
   │  └─ __init__.py
   ├─ apps.py
   ├─ core
   │  ├─ dijkstra.py
   │  ├─ distance_matrix.py
   │  ├─ ortools_optimizer.py
   │  └─ __init__.py
   ├─ migrations
   │  └─ __init__.py
   ├─ models.py
   ├─ services
   │  ├─ external_data_service.py
   │  ├─ optimization_service.py
   │  ├─ rerouting_service.py
   │  └─ __init__.py
   ├─ tests
   │  ├─ test_dijkstra.py
   │  ├─ test_optimization_service.py
   │  ├─ test_ortools_optimizer.py
   │  └─ __init__.py
   ├─ utils
   │  ├─ helpers.py
   │  └─ __init__.py
   ├─ views.py
   └─ __init__.py

```
