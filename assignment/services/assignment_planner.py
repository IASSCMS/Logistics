import logging
from typing import List
from datetime import datetime

from assignment.models.assignment import Assignment
from assignment.models.assignment_item import AssignmentItem
from assignment.services.mappers import map_vehicle_model
from fleet.models import Vehicle
from route_optimizer.services.vrp_solver import solve_cvrp
from shipments.models import Shipment
from route_optimizer.models.vrp_input import VRPInputBuilder, VRPCompiler, Location, DeliveryTask

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class AssignmentPlanner:
    def __init__(self, vehicles: List[Vehicle], shipments: List[Shipment]):
        self.vehicles = vehicles
        self.shipments = shipments

    def plan_assignments(self) -> List[Assignment]:
        logger.info("Planning assignments started.")
        builder = VRPInputBuilder()

        vehicle_map = {}
        for v in self.vehicles:
            logger.debug(f"Mapping vehicle: {v.vehicle_id}")
            mapped_vehicle = map_vehicle_model(v)
            builder.add_vehicle(mapped_vehicle)
            vehicle_map[mapped_vehicle.id] = v
        logger.info(f"{len(vehicle_map)} vehicles added to VRP input.")

        shipment_map = {}
        for s in self.shipments:
            logger.debug(f"Adding shipment: {s.shipment_id} (demand={s.demand})")
            builder.add_delivery_task(
                DeliveryTask(
                    id=str(s.id),
                    pickup=Location(lat=s.origin["lat"], lon=s.origin["lng"]),
                    delivery=Location(lat=s.destination["lat"], lon=s.destination["lng"]),
                    demand=s.demand,
                )
            )
            shipment_map[str(s.id)] = s
        logger.info(f"{len(shipment_map)} shipments added to VRP input.")

        vrp_input = VRPCompiler.compile(builder)
        logger.debug(f"Compiled VRP input with {len(vrp_input.location_ids)} locations.")

        result = solve_cvrp(vrp_input)
        logger.info("Optimizer finished solving.")

        if result["status"] != "success":
            logger.error("Optimizer failed to find a solution.")
            raise Exception("Optimization failed")

        # Implicit mapping of vehicle in this and vehicle in vrp solver
        assignments = []
        for i, route in enumerate(result["routes"]):
            vehicle = self.vehicles[i]
            logger.debug(f"Creating assignment for vehicle {vehicle.vehicle_id}, route: {route}")
            assignment = Assignment.objects.create(
                vehicle=vehicle,
                total_load=sum(
                    vrp_input.demands[node] for node in route if vrp_input.demands[node] > 0
                ),
                status='created'
            )

            # Update vehicle status, not using methods in the vehicle model but ORM directly
            vehicle.status = "assigned"
            vehicle.save(update_fields=["status"])

            seq = 1
            for node in route:
                if node in vrp_input.task_index_map:
                    task_id, role = vrp_input.task_index_map[node]
                    shipment = shipment_map[task_id]
                    loc = shipment.destination if role == "delivery" else shipment.origin

                    logger.debug(f"Adding {role} for shipment {shipment.shipment_id} at sequence {seq}")
                    AssignmentItem.objects.create(
                        assignment=assignment,
                        shipment=shipment,
                        delivery_sequence=seq,
                        delivery_location={
                            "lat": loc["lat"],
                            "lng": loc["lng"],
                        },
                        role=role
                    )
                    seq += 1

            assignments.append(assignment)

        logger.info(f"{len(assignments)} assignments successfully created.")
        return assignments
