from django.db import models

class DistanceMatrixCache(models.Model):
    """Cache for distance matrices to reduce API calls."""
    cache_key = models.CharField(max_length=255, unique=True)
    matrix_data = models.TextField()  # JSON serialized distance matrix
    location_ids = models.TextField()  # JSON serialized location IDs
    time_matrix_data = models.TextField(null=True, blank=True)  # JSON serialized time matrix
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Distance Matrix Cache"
        verbose_name_plural = "Distance Matrix Caches"
        indexes = [
            models.Index(fields=['cache_key']),
            models.Index(fields=['created_at']),
        ]
