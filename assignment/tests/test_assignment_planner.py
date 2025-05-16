from django.test import TestCase
from assignment.services.assignment_planner import AssignmentPlanner
from fleet.models import Vehicle
from shipments.models import Shipment
from assignment.models.assignment_item import AssignmentItem
import uuid


class AssignmentPlannerTestCase(TestCase):
    def setUp(self):
        self.vehicle1 = Vehicle.objects.create(
            vehicle_id="TRK001",
            capacity=100,
            status="available",
            depot_latitude=6.9,
            depot_longitude=79.8
        )

        self.vehicle2 = Vehicle.objects.create(
            vehicle_id="TRK002",
            capacity=80,
            status="available",
            depot_latitude=6.9,
            depot_longitude=79.9
        )

        self.shipment1 = Shipment.objects.create(
            shipment_id=str(uuid.uuid4())[:12],
            order_id="ORD001",
            origin={"lat": 6.9, "lng": 79.8},
            destination={"lat": 7.2, "lng": 80.6},
            demand=40,
            status="pending"
        )
        self.shipment2 = Shipment.objects.create(
            shipment_id=str(uuid.uuid4())[:12],
            order_id="ORD002",
            origin={"lat": 6.9, "lng": 79.9},
            destination={"lat": 7.3, "lng": 80.7},
            demand=30,
            status="pending"
        )
        self.shipment3 = Shipment.objects.create(
            shipment_id=str(uuid.uuid4())[:12],
            order_id="ORD003",
            origin={"lat": 7.0, "lng": 79.9},
            destination={"lat": 7.4, "lng": 80.8},
            demand=50,
            status="pending"
        )

    def test_assignments_created_successfully(self):
        planner = AssignmentPlanner(
            vehicles=[self.vehicle1, self.vehicle2],
            shipments=[self.shipment1, self.shipment2]
        )
        assignments = planner.plan_assignments()
        self.assertLessEqual(len(assignments), 2)
        self.assertGreaterEqual(AssignmentItem.objects.count(), 2)

    def test_vehicle_handles_multiple_tasks_within_capacity(self):
        planner = AssignmentPlanner(
            vehicles=[self.vehicle2],  # capacity 80
            shipments=[self.shipment1, self.shipment3]  # demands: 40 + 50 = 90 (individually valid)
        )
        assignments = planner.plan_assignments()
        self.assertEqual(len(assignments), 1)
        self.assertGreaterEqual(assignments[0].items.count(), 2)

    def test_assignment_fails_due_to_individual_task_exceeding_capacity(self):
        high_demand_1 = Shipment.objects.create(
            shipment_id="HD001",
            order_id="ORDHD1",
            origin={"lat": 6.9, "lng": 79.8},
            destination={"lat": 7.3, "lng": 80.6},
            demand=90,
            status="pending"
        )

        high_demand_2 = Shipment.objects.create(
            shipment_id="HD002",
            order_id="ORDHD2",
            origin={"lat": 7.0, "lng": 79.9},
            destination={"lat": 7.4, "lng": 80.7},
            demand=95,
            status="pending"
        )

        # vehicle2 only has 80 capacity
        planner = AssignmentPlanner(
            vehicles=[self.vehicle2],
            shipments=[high_demand_1, high_demand_2]
        )
        with self.assertRaises(Exception) as ctx:
            planner.plan_assignments()
        self.assertIn("Optimization failed", str(ctx.exception))

    def test_no_vehicles_provided(self):
        planner = AssignmentPlanner(vehicles=[], shipments=[self.shipment1])
        with self.assertRaises(AssertionError):
            planner.plan_assignments()

    def test_no_shipments_provided(self):
        planner = AssignmentPlanner(vehicles=[self.vehicle1], shipments=[])
        assignments = planner.plan_assignments()
        self.assertEqual(assignments, [])

    def test_assignments_have_delivery_items(self):
        planner = AssignmentPlanner(
            vehicles=[self.vehicle1, self.vehicle2],
            shipments=[self.shipment1, self.shipment2]
        )
        assignments = planner.plan_assignments()
        for assignment in assignments:
            self.assertGreater(
                assignment.items.count(), 0,
                f"Assignment {assignment.id} should include at least one item"
            )
