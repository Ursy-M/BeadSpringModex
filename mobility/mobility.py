
"""
This module contains fluid mobility matrices.
"""

import numpy as np
import imp

# If numba is installed import mobility_numba
try: 
  imp.find_module('numba')
  found_numba = True
except ImportError:
  found_numba = False
if found_numba:
  try:
    from . import mobility_numba
  except ImportError:
    import mobility_numba

        
        
        
def no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  vel = mobility_numba.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L)
  return vel


