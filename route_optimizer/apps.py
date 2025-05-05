from django.apps import AppConfig


class RouteOptimizerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'route_optimizer'
    verbose_name = 'Route Optimization Service'

    def ready(self):
        """
        Perform initialization tasks when the app is ready.
        """
        # Import any signals or startup tasks here
        pass
