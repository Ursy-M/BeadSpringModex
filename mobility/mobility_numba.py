
"""
Fluid mobility with numba (from Usabiaga et al(2016))
"""

import numpy as np
import scipy.sparse

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')
    
@njit(parallel=True, fastmath=True)
def no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L):
  ''' 
  Returns the product of the mobility at the bead level to the force 
  on the beads. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  force = force.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a)

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
	  
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1         

            if i == j_image:
              Mxx = fourOverThree
              Mxy = 0
              Mxz = 0
              Myy = Mxx
              Myz = 0
              Mzz = Mxx           
            else:
              # Normalize distance with hydrodynamic radius
              rx = rx * inva 
              ry = ry * inva
              rz = rz * inva
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              
              # TODO: We should not divide by zero 
              invr = 1.0 / r
              invr2 = invr * invr

              if r > 2:
                c1 = 1.0 + 2.0 / (3.0 * r2)
                c2 = (1.0 - 2.0 * invr2) * invr2
                Mxx = (c1 + c2*rx*rx) * invr
                Mxy = (     c2*rx*ry) * invr
                Mxz = (     c2*rx*rz) * invr
                Myy = (c1 + c2*ry*ry) * invr
                Myz = (     c2*ry*rz) * invr
                Mzz = (c1 + c2*rz*rz) * invr 
              else:
                c1 = fourOverThree * (1.0 - 0.28125 * r) # 9/32 = 0.28125
                c2 = fourOverThree * 0.09375 * invr      # 3/32 = 0.09375
                Mxx = c1 + c2 * rx*rx 
                Mxy =      c2 * rx*ry 
                Mxz =      c2 * rx*rz 
                Myy = c1 + c2 * ry*ry 
                Myz =      c2 * ry*rz 
                Mzz = c1 + c2 * rz*rz 
                
            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz
	  
            # 2. Compute product M_ij * F_j           
            u[i,0] += (Mxx * force[j,0] + Mxy * force[j,1] + Mxz * force[j,2]) * norm_fact_f
            u[i,1] += (Myx * force[j,0] + Myy * force[j,1] + Myz * force[j,2]) * norm_fact_f
            u[i,2] += (Mzx * force[j,0] + Mzy * force[j,1] + Mzz * force[j,2]) * norm_fact_f

  return u.flatten()        


        
