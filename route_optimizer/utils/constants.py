# Scaling factors for OR-Tools
DISTANCE_SCALING_FACTOR = 100  # Used to convert floating-point distances to integers
CAPACITY_SCALING_FACTOR = 100  # Used to convert floating-point capacities to integers
TIME_SCALING_FACTOR = 60       # Used to convert minutes to seconds for time windows

# Bounds for valid distance values
MAX_SAFE_DISTANCE = 1e6        # Maximum safe distance value (km)
MIN_SAFE_DISTANCE = 0.0        # Minimum safe distance value (km)

# Bounds for valid time values
MAX_SAFE_TIME = 24 * 60        # Maximum safe time value (minutes) - 24 hours
MIN_SAFE_TIME = 0.0            # Minimum safe time value (minutes)

# --- Delivery Priorities ---
# Define your priority levels here.
# Lower integer might mean higher priority, or vice versa, depending on your logic.
# Example: Higher number = higher priority
PRIORITY_LOW = 1
PRIORITY_NORMAL = 2  # Or PRIORITY_MEDIUM
PRIORITY_HIGH = 3
PRIORITY_URGENT = 4

# Default priority if not specified
DEFAULT_DELIVERY_PRIORITY = PRIORITY_NORMAL

