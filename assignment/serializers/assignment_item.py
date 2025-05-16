from rest_framework import serializers
from assignment.models.assignment_item import AssignmentItem
from shipments.models import Shipment


class ShipmentSerializerForAssignment(serializers.ModelSerializer):
    class Meta:
        model = Shipment
        fields = ['id', 'order_id', 'demand', 'status']  # Add more as needed

class AssignmentItemSerializer(serializers.ModelSerializer):
    shipment = ShipmentSerializerForAssignment(read_only=True)

    class Meta:
        model = AssignmentItem
        fields = ['shipment', 'role', 'delivery_sequence', 'delivery_location', 'is_delivered', 'delivered_at']
