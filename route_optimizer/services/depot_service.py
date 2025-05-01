class DepotService:
    @staticmethod
    def find_depot_index(locations):
        depots = [i for i, loc in enumerate(locations) if loc.is_depot]
        return depots[0] if depots else 0
