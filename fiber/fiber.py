
"""
Small class to handle a single fiber.
"""

import numpy as np


class Fiber(object):
    
    def __init__(self, fiber_bead_locations, centerline_distance_factor, bead_fiber_radius):
        
        # number of fiber beads
        self.NF = len(fiber_bead_locations)
        # fiber bead radius
        self.bead_fiber_radius = bead_fiber_radius
        # fiber bead radius array
        self.bead_fiber_radius_array = np.ones((self.NF), dtype=float) * self.bead_fiber_radius
        # centerline distance
        self.centerline_distance = centerline_distance_factor * self.bead_fiber_radius
        # length
        self.fiber_length = self.centerline_distance * self.NF
        # location
        self.fiber_bead_locations = np.copy(fiber_bead_locations)
        # fiber ID
        self.ID = None
        
          
    def get_fiber_r_vectors(self, fiber_bead_locations=None):
        """
        Parameters
        ----------
        fiber_bead_locations : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        fiber_r_vectors : TYPE
            DESCRIPTION.

        """
        if fiber_bead_locations is None:
            fiber_bead_locations = self.fiber_bead_locations
        
        fiber_r_vectors = fiber_bead_locations
        
        return  fiber_r_vectors
    
    
    
    
        
        
        
    