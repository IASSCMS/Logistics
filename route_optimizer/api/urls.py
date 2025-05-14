"""
URL configuration for the route optimizer API.

This module defines the URL patterns for the route optimization API endpoints.
"""
from django.urls import path
from route_optimizer.api.views import OptimizeRoutesView, RerouteView, health_check

app_name = 'route_optimizer'

urlpatterns = [
    # Health check endpoint
    path('health/', health_check, name='health_check_get'),
    
    # Route optimization endpoints
    path('optimize/', OptimizeRoutesView.as_view(), name='optimize_routes_create'),
    path('reroute/', RerouteView.as_view(), name='reroute_vehicles_update'),
]