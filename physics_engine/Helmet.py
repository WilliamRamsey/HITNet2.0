class Helmet:
    def __init__(self, IDs = [], masks = [], collisions = []):
        self.IDs = IDs # List of IDs that helmet mask is appears in image
        self.masks = masks # List of masks -> Should we transform or not?
        self.collisions = collisions # List of indexes where helmet experienced a collision
