
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


@njit(parallel=True, fastmath=True)
def mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, L, wall):
  '''
  Flow created on target beads by force applied on source beads. 
  beads can have different radius.
  '''
  # Prepare vectors
  num_targets = target.size // 3
  num_sources = source.size // 3
  source = source.reshape(num_sources, 3)
  target = target.reshape(num_targets, 3)
  force = force.reshape(num_sources, 3)
  u = np.zeros((num_targets, 3))
  fourOverThree = 4.0 / 3.0
  norm_fact_f = 1.0 / (8.0 * np.pi * eta)
  
  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0
  periodic_z = 0 
  Lx = L[0]
  Ly = L[1]
  Lz = L[2]
  if Lx > 0:
    periodic_x = 1
  if Ly > 0:
    periodic_y = 1
  if Lz > 0:
    periodic_z = 1

  # Copy to one dimensional vectors
  rx_src = np.copy(source[:,0])
  ry_src = np.copy(source[:,1])
  rz_src = np.copy(source[:,2])
  rx_trg = np.copy(target[:,0])
  ry_trg = np.copy(target[:,1])
  rz_trg = np.copy(target[:,2])
  fx_vec = np.copy(force[:,0])
  fy_vec = np.copy(force[:,1])
  fz_vec = np.copy(force[:,2])

  # Loop over image boxes and then over particles
  for i in prange(num_targets):
    rxi = rx_trg[i]
    ryi = ry_trg[i]
    rzi = rz_trg[i]
    a = radius_target[i]
    ux, uy, uz = 0, 0, 0
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(num_sources):
            b = radius_source[j]
            
            # Compute vector between particles i and j
            rx = rxi - rx_src[j]
            ry = ryi - ry_src[j]
            rz = rzi - rz_src[j]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if Lx > 0:
              rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
              rx = rx + boxX * Lx
            if Ly > 0:
              ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
              ry = ry + boxY * Ly 
            if Lz > 0:
              rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz
              rz = rz + boxZ * Lz            

            # Compute interaction without wall
            r2 = rx*rx + ry*ry + rz*rz
            r = np.sqrt(r2)
            
            if r > (a + b):
              a2 = a * a
              b2 = b * b
              C1 = (1 + (b2+a2) / (3 * r2)) / r
              C2 = ((1 - (b2+a2) / r2) / r2) / r
            elif r > abs(b-a):
              r3 = r2 * r
              C1 = ((16*(b+a)*r3 - np.power(np.power(b-a,2) + 3*r2,2)) / (32*r3)) * fourOverThree / (b * a)
              C2 = ((3*np.power(np.power(b-a,2)-r2, 2) / (32*r3)) / r2) * fourOverThree / (b * a)
            else:
              largest_radius = a if a > b else b
              C1 = fourOverThree / largest_radius
              C2 = 0             

            Mxx = C1 + C2 * rx * rx;
            Mxy =      C2 * rx * ry;
            Mxz =      C2 * rx * rz;
            Myy = C1 + C2 * ry * ry;
            Myz =      C2 * ry * rz;
            Mzz = C1 + C2 * rz * rz;
            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz

            # If wall compute correction
            if wall:
              y3 = rz_src[j]
              x3 = rzi
              rz = rzi + rz_src[j]
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              a2 = a * a
              b2 = b * b
              r3 = r2 * r
              r5 = r3 * r2
              r7 = r5 * r2
              r9 = r7 * r2
       
              Mxx -= ((1+(b2+a2)/(3.0*r2)) + (1-(b2+a2)/r2) * rx * rx / r2) / r
              Mxy -= (                       (1-(b2+a2)/r2) * rx * ry / r2) / r
              Mxz += (                       (1-(b2+a2)/r2) * rx * rz / r2) / r
              Myx -= (                       (1-(b2+a2)/r2) * ry * rx / r2) / r
              Myy -= ((1+(b2+a2)/(3.0*r2)) + (1-(b2+a2)/r2) * ry * ry / r2) / r
              Myz += (                       (1-(b2+a2)/r2) * ry * rz / r2) / r
              Mzx -= (                       (1-(b2+a2)/r2) * rz * rx / r2) / r
              Mzy -= (                       (1-(b2+a2)/r2) * rz * ry / r2) / r
              Mzz += ((1+(b2+a2)/(3.0*r2)) + (1-(b2+a2)/r2) * rz * rz / r2) / r

              # M[l][m] += 2*(-J[l][m]/r - r[l]*x3[m]/r3 - y3[l]*r[m]/r3 + x3*y3*(I[l][m]/r3 - 3*r[l]*r[m]/r5))
              Mxx -= 2*(x3*y3*(1.0/r3 - 3*rx*rx/r5))
              Mxy -= 2*(x3*y3*(       - 3*rx*ry/r5))
              Mxz += 2*(-rx*x3/r3 + x3*y3*( -3*rx*rz/r5))
              Myx -= 2*(x3*y3*(       - 3*ry*rx/r5))
              Myy -= 2*(x3*y3*(1.0/r3 - 3*ry*ry/r5))
              Myz += 2*(-ry*x3/r3 + x3*y3*( -3*ry*rz/r5))
              Mzx -= 2*(-y3*rx/r3 + x3*y3*( -3*rz*rx/r5))
              Mzy -= 2*(-y3*ry/r3 + x3*y3*( -3*rz*ry/r5))
              Mzz += 2*(-1.0/r - rz*x3/r3 - y3*rz/r3 + x3*y3*(1.0/r3 - 3*rz*rz/r5))
              
              # M[l][m] += (2*a2/3.0) * (-J[l][m]/r3 + 3*r[l]*rz[m]/r5 - y3*(3*rz*I[l][m]/r5 + 3*delta_3[l]*r[m]/r5 + 3*r[l]*delta_3[m]/r5 - 15*rz*r[l]*r[m]/r7))
              Mxx -= (2*a2/3.0) * (-y3*(3*rz/r5 - 15*rz*rx*rx/r7))
              Mxy -= (2*a2/3.0) * (-y3*(        - 15*rz*rx*ry/r7))
              Mxz += (2*a2/3.0) * (3*rx*rz/r5 - y3*(3*rx/r5 - 15*rz*rx*rz/r7))
              Myx -= (2*a2/3.0) * (-y3*(        - 15*rz*ry*rx/r7))
              Myy -= (2*a2/3.0) * (-y3*(3*rz/r5 - 15*rz*ry*ry/r7))
              Myz += (2*a2/3.0) * (3*ry*rz/r5 - y3*(3*ry/r5 - 15*rz*ry*rz/r7))
              Mzx -= (2*a2/3.0) * (-y3*(3*rx/r5 - 15*rz*rz*rx/r7))
              Mzy -= (2*a2/3.0) * (-y3*(3*ry/r5 - 15*rz*rz*ry/r7))
              Mzz += (2*a2/3.0) * (-1.0/r3 + 3*rz*rz/r5 - y3*(3*rz/r5 + 3*rz/r5 + 3*rz/r5 - 15*rz*rz*rz/r7))

              # M[l][m] += (2*b2/3.0) * (-J[l][m]/r3 + 3*rz[l]*r[m]/r5 - x3*(3*rz*I[l][m]/r5 + 3*delta_3[l]*r[m]/r5 + 3*r[l]*delta_3[m]/r5 - 15*rz*r[l]*r[m]/r7))
              Mxx -= (2*b2/3.0) * (-x3*(3*rz/r5 - 15*rz*rx*rx/r7))
              Mxy -= (2*b2/3.0) * (-x3*(        - 15*rz*rx*ry/r7))
              Mxz += (2*b2/3.0) * (-x3*(3*rx/r5 - 15*rz*rx*rz/r7))
              Myx -= (2*b2/3.0) * (-x3*(        - 15*rz*ry*rx/r7))
              Myy -= (2*b2/3.0) * (-x3*(3*rz/r5 - 15*rz*ry*ry/r7))
              Myz += (2*b2/3.0) * (-x3*(3*ry/r5 - 15*rz*ry*rz/r7))
              Mzx -= (2*b2/3.0) * (3*rz*rx/r5 - x3*(3*rx/r5 - 15*rz*rz*rx/r7))
              Mzy -= (2*b2/3.0) * (3*rz*ry/r5 - x3*(3*ry/r5 - 15*rz*rz*ry/r7))
              Mzz += (2*b2/3.0) * (-1.0/r3 + 3*rz*rz/r5 - x3*(3*rz/r5 + 3*rz/r5 + 3*rz/r5 - 15*rz*rz*rz/r7))

              # M[l][m] += (2*b2*a2/3.0) * (-I[l][m]/r5 + 5*rz*rz*I[l][m]/r7 - J[l][m]/r5 + 5*rz[l]*r[m]/r7 - J[l][m]/r5 + 5*r[l]*rz[m]/r7 + 5*rz[l]*r[m]/r7 + 5*r[l]*r[m]/r7 + 5*r[l]*rz[m]/r7 - 35 * rz*rz*r[l]*r[m]/r9)
              Mxx -= (2*b2*a2/3.0) * (-1.0/r5 + 5*rz*rz/r7 + 5*rx*rx/r7 - 35 * rz*rz*rx*rx/r9)
              Mxy -= (2*b2*a2/3.0) * (          5*rx*ry/r7 +            - 35 * rz*rz*rx*ry/r9)
              Mxz += (2*b2*a2/3.0) * (5*rx*rz/r7 + 5*rx*rz/r7 + 5*rx*rz/r7 - 35 * rz*rz*rx*rz/r9)
              Myx -= (2*b2*a2/3.0) * (5*ry*rx/r7 - 35 * rz*rz*ry*rx/r9)
              Myy -= (2*b2*a2/3.0) * (-1.0/r5 + 5*rz*rz/r7 + 5*ry*ry/r7 - 35 * rz*rz*ry*ry/r9)
              Myz += (2*b2*a2/3.0) * (5*ry*rz/r7 + 5*ry*rz/r7 + 5*ry*rz/r7 - 35 * rz*rz*rz*ry/r9)
              Mzx -= (2*b2*a2/3.0) * (5*rz*rx/r7 + 5*rz*rx/r7 + 5*rz*rx/r7 - 35 * rz*rz*rz*rx/r9)
              Mzy -= (2*b2*a2/3.0) * (5*rz*ry/r7 + 5*rz*ry/r7 + 5*rz*ry/r7 - 35 * rz*rz*rz*ry/r9)
              Mzz += (2*b2*a2/3.0) * (-1.0/r5 + 5*rz*rz/r7 - 1.0/r5 + 5*rz*rz/r7 - 1.0/r5 + 5*rz*rz/r7 + 5*rz*rz/r7 + 5*rz*rz/r7 + 5*rz*rz/r7 - 35 * rz*rz*rz*rz/r9)
              
            # 2. Compute product M_ij * F_j           
            ux += (Mxx * fx_vec[j] + Mxy * fy_vec[j] + Mxz * fz_vec[j]) * norm_fact_f
            uy += (Myx * fx_vec[j] + Myy * fy_vec[j] + Myz * fz_vec[j]) * norm_fact_f
            uz += (Mzx * fx_vec[j] + Mzy * fy_vec[j] + Mzz * fz_vec[j]) * norm_fact_f

    u[i,0] = ux
    u[i,1] = uy
    u[i,2] = uz

  return u.flatten()

        
