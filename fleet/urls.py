from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    VehicleViewSet, MaintenanceRecordViewSet, 
    FuelRecordViewSet, TripRecordViewSet
)

router = DefaultRouter()
router.register(r'vehicles', VehicleViewSet)
router.register(r'maintenance', MaintenanceRecordViewSet)
router.register(r'fuel', FuelRecordViewSet)
router.register(r'trips', TripRecordViewSet)

urlpatterns = [
    path('', include(router.urls)),
]