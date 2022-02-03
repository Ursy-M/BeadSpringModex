

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


def no_wall_mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the beads.
  Mobility for particles in unbounded domain.  
  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.
  For beads overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  This function uses numba.
  '''
  # Get domain size for Pseudo-PBC
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Compute M_tilde * B * force
  velocities = mobility_numba.mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, L=L, wall=0)

  return velocities


def no_wall_mobility_trans_times_force_source_target(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.
  Compute velocity of targets of radius radius_target due
  to forces on sources of radius source_targer in unbounded domain.
  That is, compute the matrix vector product
  velocities_target = M_tt * forces_sources
  where M_tt has dimensions (target, source)
  See Reference P. J. Zuk et al. J. Fluid Mech. (2014), vol. 741, R5, doi:10.1017/jfm.2013.668
  '''
  force = np.reshape(force, (force.size // 3, 3))
  velocity = np.zeros((target.size // 3, 3))
  prefactor = 1.0 / (8 * np.pi * eta)
  b2 = radius_target**2
  a2 = radius_source**2
  # Loop over targets
  for i, r_target in enumerate(target):
    # Distance between target and sources
    r_source_to_target = r_target - source
    # Loop over sources
    for j, r in enumerate(r_source_to_target):
      r2 = np.dot(r,r)
      r_norm  = np.sqrt(r2)
      # Compute 3x3 block mobility
      if r_norm >= (radius_target[i] + radius_source[j]):
        Mij = (1 + (b2[i]+a2[j]) / (3 * r2)) * np.eye(3) + (1 - (b2[i]+a2[j]) / r2) * np.outer(r,r) / r2
        Mij = (prefactor / r_norm) * Mij
      elif r_norm > np.absolute(radius_target[i]-radius_source[j]):
        r3 = r_norm * r2
        Mij = ((16*(radius_target[i]+radius_source[j])*r3 - ((radius_target[i]-radius_source[j])**2 + 3*r2)**2) / (32*r3)) * np.eye(3) +\
            ((3*((radius_target[i]-radius_source[j])**2-r2)**2) / (32*r3)) * np.outer(r,r) / r2
        Mij = Mij / (6 * np.pi * eta * radius_target[i] * radius_source[j])
      else:
        largest_radius = radius_target[i] if radius_target[i] > radius_source[j] else radius_source[j]
        Mij = (1.0 / (6 * np.pi * eta * largest_radius)) * np.eye(3)
      velocity[i] += np.dot(Mij, force[j])

  return velocity
