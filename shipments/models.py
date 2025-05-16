from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError


class Shipment(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('scheduled', 'Scheduled'),
        ('dispatched', 'Dispatched'),
        ('in_transit', 'In Transit'),
        ('delivered', 'Delivered'),
        ('failed', 'Failed'),
    ]

    shipment_id = models.CharField(max_length=32, unique=True)
    order_id = models.CharField(max_length=32)  # Reference to order service

    origin = models.JSONField()
    destination = models.JSONField()

    demand = models.PositiveIntegerField(help_text="Amount of load required for this shipment (e.g., in kg or units)", default=0)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    scheduled_dispatch = models.DateTimeField(null=True, blank=True)
    actual_dispatch = models.DateTimeField(null=True, blank=True)
    delivery_time = models.DateTimeField(null=True, blank=True)
    assigned_vehicle_id = models.CharField(max_length=36, null=True, blank=True)  # Optional

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def mark_pending(self):
        if self.status not in ['scheduled', 'failed']:
            raise ValidationError("Can only revert to pending from scheduled or failed.")
        self.status = 'pending'
        self.save(update_fields=['status', 'updated_at'])

    def mark_scheduled(self, scheduled_time=None):
        if self.status != 'pending':
            raise ValidationError("Can only schedule a shipment that is pending.")
        self.status = 'scheduled'
        if scheduled_time:
            self.scheduled_dispatch = scheduled_time
        self.save(update_fields=['status', 'scheduled_dispatch', 'updated_at'])

    def mark_dispatched(self, dispatch_time=None):
        if self.status != 'scheduled':
            raise ValidationError("Can only dispatch a shipment that is scheduled.")
        self.status = 'dispatched'
        self.actual_dispatch = dispatch_time or timezone.now()
        self.save(update_fields=['status', 'actual_dispatch', 'updated_at'])

    def mark_in_transit(self):
        if self.status != 'dispatched':
            raise ValidationError("Can only mark in transit after dispatch.")
        self.status = 'in_transit'
        self.save(update_fields=['status', 'updated_at'])

    def mark_delivered(self, delivery_time=None):
        if self.status != 'in_transit':
            raise ValidationError("Can only mark delivered after in_transit.")
        self.status = 'delivered'
        self.delivery_time = delivery_time or timezone.now()
        self.save(update_fields=['status', 'delivery_time', 'updated_at'])

    def mark_failed(self):
        if self.status not in ['pending', 'scheduled', 'dispatched', 'in_transit']:
            raise ValidationError("Only active shipments can be marked as failed.")
        self.status = 'failed'
        self.save(update_fields=['status', 'updated_at'])

    def __str__(self):
        return f"{self.shipment_id} ({self.status})"
