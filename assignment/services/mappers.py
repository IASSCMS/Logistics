from route_optimizer.models.vrp_input import Vehicle as VRPVehicle, Location

def map_vehicle_model(vehicle_model):
    if vehicle_model.depot_latitude is None or vehicle_model.depot_longitude is None:
        raise ValueError(f"Vehicle {vehicle_model.vehicle_id} missing depot coordinates")

    return VRPVehicle(
        id=vehicle_model.vehicle_id,
        capacity=vehicle_model.capacity,
        depot=Location(
            lat=float(vehicle_model.depot_latitude),
            lon=float(vehicle_model.depot_longitude)
        )
    )
