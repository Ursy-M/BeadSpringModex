"""
Module which contains steric forces implemented with numba.
"""

import numpy as np

# try to import numba
try:
    from numba import njit, prange
except ImportError:
    print('numba not found')
    

@njit(parallel=True, fastmath=True)
def steric_forces_numba_kernel(r_vectors, L, eps, gamma, a_array, khi, bead_list_index):
    """
    Parameters
    ----------
    r_vectors : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    eps : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    a_array : TYPE
        DESCRIPTION.
    khi : TYPE
        DESCRIPTION.
    bead_list_index : TYPE
        DESCRIPTION.

    Returns
    -------
    force : TYPE
        DESCRIPTION.

    """
    
    # get number of beads
    number_of_beads = r_vectors.size // 3
    # set the vector force
    force = np.zeros((number_of_beads, 3))
    
    # double loop over beads to compute steric forces
    for i in prange(number_of_beads):
        for j in range(number_of_beads):
            iw = bead_list_index[i]
            jw = bead_list_index[j]
            if i==j:
                continue
            elif abs(i-j)==1 and iw==jw:
                continue
            
            dr = np.zeros(3)
            for k in range(3):
                # compute center - center vector
                dr[k] = r_vectors[j, k] - r_vectors[i, k]
                # use distance with pseudo-periodic boundaries conditions
                if L[k] > 0:
                    dr[k] -= int(dr[k] / L[k] + 0.5 * (int(dr[k]>0) - int(dr[k]<0))) * L[k] 
            
            # compute forces
            r_norm = np.linalg.norm(dr)
            radii_sum = a_array[i] + a_array[j]
            r_ref = khi * radii_sum
            if r_norm > r_ref:
                factor = 0.
            elif r_norm < r_ref:
                # compute the prefactors
                xi_0 = r_ref**2 - r_norm**2
                xi_1 = r_ref**2 - radii_sum**2
                factor = -(eps/radii_sum) * (xi_0/xi_1)**(2*gamma)
            
            for k in range(3):
                force[i, k] += factor * dr[k]
                
    return force


def calc_steric_forces_numba(r_vectors, *args, **kwargs):
    """
    This function computes the steric interactions.
    """
    # get parameters from entries
    L = kwargs.get('periodic_length')
    eps = kwargs.get('repulsion_strength')
    gamma = kwargs.get('stiffness_parameter')
    a_array = kwargs.get('radius_array')
    khi = kwargs.get('contact_distance_factor')
    bead_list_index = kwargs.get('bead_list_index')
    
    # compute forces
    forces = steric_forces_numba_kernel(r_vectors, L, eps, gamma, a_array, khi, bead_list_index)
    
    return forces
    