from django.db import models
from assignment.models.assignment import Assignment
from shipments.models import Shipment


class AssignmentItem(models.Model):
    ROLE_CHOICES = [
        ("pickup", "Pickup"),
        ("delivery", "Delivery"),
    ]

    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name='items')
    shipment = models.ForeignKey(Shipment, on_delete=models.CASCADE)
    delivery_sequence = models.PositiveIntegerField()  # 1st, 2nd, 3rd stop, etc.
    delivery_location = models.JSONField()  # { "lat": ..., "lng": ... }

    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default="delivery")  # NEW

    # TODO: Consider renaming 'is_delivered' and 'delivered_at' for better clarity. 
    #       Example: 'is_delivered' -> 'has_been_delivered', 'delivered_at' -> 'delivery_timestamp'.
    is_delivered = models.BooleanField(default=False)
    delivered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['delivery_sequence']

    def __str__(self):
        return f"{self.role.capitalize()} for Shipment {self.shipment.id} in Assignment {self.assignment.id}"
