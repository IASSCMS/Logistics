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


# # Scaling factors for optimization algorithm
# DISTANCE_SCALING_FACTOR = 1000  # Convert km to meters for integer calculations
# CAPACITY_SCALING_FACTOR = 100   # Scale capacity values for integer calculations
# TIME_SCALING_FACTOR = 60        # Convert minutes to seconds

# # Safety limits
# MAX_SAFE_DISTANCE = 10000.0     # Maximum reasonable distance in km
# MAX_SAFE_TIME = 24 * 60 * 60    # Maximum reasonable time in seconds (24 hours)
