from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# === DTOs ===

@dataclass(frozen=True)
class Location:
    lat: float
    lon: float


@dataclass
class Vehicle:
    id: str
    depot: Location
    capacity: int


@dataclass
class DeliveryTask:
    id: str
    pickup: Location
    delivery: Location
    demand: int


# === VRP Input Model ===

@dataclass
class VRPInput:
    location_ids: List[str]
    distance_matrix: List[List[int]]
    starts: List[int]
    ends: List[int]
    vehicle_capacities: List[int]
    num_vehicles: int
    task_index_map: Dict[int, Tuple[str, str]]  # (task_id, "pickup"/"delivery")
    demands: List[int]
    time_limit: int = 3
    pickups_deliveries: List[Tuple[int, int]] = field(default_factory=list)
    vehicles: List[Vehicle] = field(default_factory=list)

    location_id_to_index: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.location_id_to_index = {loc_id: i for i, loc_id in enumerate(self.location_ids)}

    def validate(self):
        n = len(self.location_ids)
        assert len(self.demands) == n, "Mismatch: demands vs location_ids"
        assert len(self.distance_matrix) == n, "Mismatch: matrix rows vs location_ids"
        assert len(self.vehicles)  > 0, "No vehicles defined"
        assert len(self.vehicles) == self.num_vehicles, "Mismatch: vehicles vs num_vehicles"
        assert all(len(row) == n for row in self.distance_matrix), "Matrix must be square"
        for v in self.vehicles:
            depot_index = self.location_id_to_index.get(f"{v.id}_depot")
            assert depot_index is not None, f"Depot location for vehicle {v.id} not found"


# === Builder ===

class VRPInputBuilder:
    def __init__(self):
        self.locations: List[Location] = []
        self.distance_matrix: List[List[int]] = []
        self.vehicles: List[Vehicle] = []
        self.tasks: List[DeliveryTask] = []
        self.location_labels: List[str] = []

    def _add_location(self, loc: Location, label: str) -> int:
        if label in self.location_labels:
            raise ValueError(f"Duplicate label detected: {label}")
        index = len(self.locations)
        self.locations.append(loc)
        self.location_labels.append(label)
        for row in self.distance_matrix:
            row.append(0)
        self.distance_matrix.append([0] * (index + 1))
        return index

    def set_distance(self, from_index: int, to_index: int, distance: int):
        self.distance_matrix[from_index][to_index] = distance
        self.distance_matrix[to_index][from_index] = distance

    def add_vehicle(self, vehicle: Vehicle):
        self._add_location(vehicle.depot, f"{vehicle.id}_depot")
        self.vehicles.append(vehicle)

    def add_delivery_task(self, task: DeliveryTask):
        self._add_location(task.pickup, f"{task.id}_pickup")
        self._add_location(task.delivery, f"{task.id}_delivery")
        self.tasks.append(task)


# === Compiler ===

class VRPCompiler:
    @staticmethod
    def compile(builder: VRPInputBuilder) -> VRPInput:
        location_ids = builder.location_labels.copy()
        label_to_index = {label: i for i, label in enumerate(location_ids)}

        demands = [0] * len(builder.locations)
        deliveries = []
        task_index_map: Dict[int, Tuple[str, str]] = {}

        for task in builder.tasks:
            pickup_label = f"{task.id}_pickup"
            delivery_label = f"{task.id}_delivery"
            pickup_idx = label_to_index[pickup_label]
            delivery_idx = label_to_index[delivery_label]

            demands[pickup_idx] += task.demand
            demands[delivery_idx] -= task.demand
            deliveries.append((pickup_idx, delivery_idx))
            task_index_map[pickup_idx] = (task.id, "pickup")
            task_index_map[delivery_idx] = (task.id, "delivery")

        starts, ends = [], []
        for v in builder.vehicles:
            depot_label = f"{v.id}_depot"
            depot_idx = label_to_index[depot_label]
            starts.append(depot_idx)
            ends.append(depot_idx)

        return VRPInput(
            location_ids=location_ids,
            distance_matrix=builder.distance_matrix,
            demands=demands,
            vehicle_capacities=[v.capacity for v in builder.vehicles],
            num_vehicles=len(builder.vehicles),
            starts=starts,
            ends=ends,
            pickups_deliveries=deliveries,
            task_index_map=task_index_map,
            vehicles=builder.vehicles
        )
