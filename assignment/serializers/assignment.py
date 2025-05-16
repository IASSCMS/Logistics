from rest_framework import serializers
from assignment.models.assignment import Assignment
from assignment.serializers.assignment_item import AssignmentItemSerializer

class AssignmentSerializer(serializers.ModelSerializer):
    items = AssignmentItemSerializer(many=True, read_only=True)
    vehicle = serializers.CharField(source='vehicle.vehicle_id', read_only=True)

    class Meta:
        model = Assignment
        fields = ['id', 'vehicle', 'total_load', 'status', 'items']