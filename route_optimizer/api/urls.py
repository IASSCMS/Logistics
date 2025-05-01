"""
URL configuration for the route optimizer API.

This module defines the URL patterns for the route optimization API endpoints.
"""
from django.urls import path
from route_optimizer.api.views import OptimizeRoutesView, RerouteView, health_check

app_name = 'route_optimizer'

urlpatterns = [
    # Health check endpoint
    path('health/', health_check, name='health_check'),
    
    # Route optimization endpoints
    path('optimize/', OptimizeRoutesView.as_view(), name='optimize_routes'),
    path('reroute/', RerouteView.as_view(), name='reroute'),
]