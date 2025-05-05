from route_optimizer.core.ortools_optimizer import Delivery

def shipment_to_pickup_and_delivery(shipment):
    return [
        Delivery(
            id=f"{shipment.shipment_id}_P",
            location_id=shipment.origin_warehouse_id,
            demand=1.0,
            is_pickup=True
        ),
        Delivery(
            id=f"{shipment.shipment_id}_D",
            location_id=shipment.destination_warehouse_id,
            demand=1.0,
            is_pickup=False
        )
    ]

def map_shipments_to_deliveries(shipments):
    deliveries = []
    pickup_delivery_pairs = []

    for shipment in shipments:
        pickup, delivery = shipment_to_pickup_and_delivery(shipment)
        deliveries.extend([pickup, delivery])
        pickup_delivery_pairs.append((pickup.id, delivery.id))  # used in optimizer

    return deliveries, pickup_delivery_pairs
