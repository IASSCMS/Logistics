from django.contrib import admin
from .models import Shipment


@admin.register(Shipment)
class ShipmentAdmin(admin.ModelAdmin):
    list_display = (
        'shipment_id',
        'order_id',
        'get_origin',
        'get_destination',
        'demand',
        'status',
        'created_at',
    )
    list_filter = ('status',)
    search_fields = ('shipment_id', 'order_id')
    ordering = ('-created_at',)

    @admin.display(description="Origin")
    def get_origin(self, obj):
        loc = obj.origin
        return f"{loc.get('lat')}, {loc.get('lng')}" if loc else "N/A"

    @admin.display(description="Destination")
    def get_destination(self, obj):
        loc = obj.destination
        return f"{loc.get('lat')}, {loc.get('lng')}" if loc else "N/A"
