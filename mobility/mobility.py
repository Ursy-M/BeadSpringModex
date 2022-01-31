
"""
This module contains fluid mobility matrices (from Usabiaga et al(2016)).
"""

import numpy as np
import imp


# if pycuda is installed import mobility_pycuda
try:
    imp.find_module('pycuda')
    found_pycuda = True
except ImportError:
    found_pycuda = False
if found_pycuda:
    try:
        import pycuda.autoinit
        autoinit_pycuda = True
    except:
        autoinit_pycuda = False
    if autoinit_pycuda:
        try:
            from . import mobility_pycuda
        except ImportError:
            from .mobility import mobility_pycuda

# if numba is installed import mobility_numba
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
  Returns the product of the mobility at the bead level to the force 
  on the beads. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  vel = mobility_numba.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L)
  return vel

def no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the bead level to the force
  on the beads. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses pycuda.
  '''
  vel = mobility_pycuda.no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs)
  return vel
