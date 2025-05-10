class DepotService:
    def get_nearest_depot(self, locations):
        """
        Find the nearest depot from the given locations.
        
        Args:
            locations: List of Location objects.
            
        Returns:
            Location object representing the depot.
            If no depot is found, returns the first location.
        """
        # Find locations marked as depots
        depots = [loc for loc in locations if getattr(loc, 'is_depot', False)]
        
        # If no depots found, use the first location as default
        if not depots and locations:
            return locations[0]
        elif not depots:
            # No locations at all
            return None
        
        # If only one depot, return it
        if len(depots) == 1:
            return depots[0]
        
        # If multiple depots, we could implement logic to find the most central one
        # For now, just return the first depot
        return depots[0]
    
    @staticmethod
    def find_depot_index(locations):
        """
        Find the index of the depot in the locations list.
        
        Args:
            locations: List of Location objects.
            
        Returns:
            Index of the depot in the locations list.
            If no depot is found, returns 0.
        """
        for i, location in enumerate(locations):
            if getattr(location, 'is_depot', False):
                return i
        
        # Default to the first location if no depot is marked
        return 0
    # @staticmethod
    # def find_depot_index(locations):
    #     depots = [i for i, loc in enumerate(locations) if loc.is_depot]
    #     return depots[0] if depots else 0
