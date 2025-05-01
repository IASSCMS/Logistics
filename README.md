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


