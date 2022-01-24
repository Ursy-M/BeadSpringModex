
"""
Module which contains functions to define forces applied on the beads.
"""

import numpy as np


# import project functions and modules
from fiber import fiber


def set_steric_forces(implementation):
    """
    set the function to compute the bead-bead forces to the right function.
    the implementation on numba and pycuda are much faster than the 
    implementation on python.
    """
    
    if implementation == 'None':
        return defaut_zero_r_vectors
    elif implementation == 'python':
        return calc_steric_forces_python
    elif implementation == 'numba':
        return calc_steric_forces_numba
    elif implementation == 'pycuda':
        return calc_steric_forces_pycuda
    

def project_to_periodic_image(r, L):
    """
    project a vector r to the minimal image representation centered around
    (0, 0, 0) and of size L = (Lx, Ly, Lz). If any dimension of L is equal 
    or smaller than zero the box is assumed to be infinite in that direction.
    
    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if L is not None:
        for i in range(3):
            if L[i] > 0:
                r[i] = r[i] - int(r[i] / L[i] + 0.5 * (int(r[i] > 0) - int(r[i] < 0))) * L[i]
    return r


def defaut_zero_r_vectors(r_vectors, *args, **kwargs):
    """
    Parameters
    ----------
    r_vectors : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    return np.zeros((r_vectors//3, 3))


def bead_bead_force(r, ind_i, ind_j, *args, **kwargs):
    """
    this function compute the force between two beads with vector between
    bead centers r.
    
    the force is computed from the following formula:
    
    
    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    ind_i : TYPE
        DESCRIPTION.
    ind_j : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # get parameters form entries
    L = kwargs.get('periodic_length')
    eps = kwargs.get('repulsion_strength')
    gamma = kwargs.get('stiffness_parameter')
    a_array = kwargs.get('radius_array')
    khi = kwargs.get('contact_distance_factor')
    
    # compute force
    project_to_periodic_image(r, L)
    r_norm = np.linalg.norm(r)
    radii_sum = a_array[ind_i] + a_array[ind_j]
    r_ref = khi * radii_sum
    if r_norm > r_ref:
        return 0.0
    elif r_norm < r_ref:
        # compute the prefactors
        xi_0 = r_ref**2 - r_norm**2
        xi_1 = r_ref**2 - radii_sum**2
        return (-(eps/radii_sum) * (xi_0/xi_1)**(2*gamma)) * r


def calc_steric_forces_python(r_vectors, *args, **kwargs):
    """
    this function computes the steric interactions.
    
    Parameters
    ----------
    r_vectors : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    forces : TYPE
        DESCRIPTION.

    """
    # get number of beads
    number_of_beads = r_vectors.size // 3
    # set the vector force
    forces = np.zeros((number_of_beads, 3))
    
    # get parameters from entries
    bead_list_index = kwargs.get('bead_list_index')
    
    # double loop over beads to compute steric forces
    for i  in range(number_of_beads):
        for j in range(number_of_beads):
            iw = bead_list_index[i]
            jw = bead_list_index[j]
            if i==j:
                force = 0.0
            elif abs(i-j)==1 and iw==jw:
                force = 0.0
            else:
                # compute the center - center vector
                r = r_vectors[j] - r_vectors[i]
                force = bead_bead_force(r, i, j, *args, **kwargs)
            forces[i] += force
            forces[j] -= force
    
    return forces

    
def calc_steric_forces_numba(r_vectors, *args, **kwargs):
    
    # get number of beads
    number_of_beads = r_vectors.size // 3
    # set the vector force
    forces = np.zeros((number_of_beads, 3))
    
    return forces


def calc_steric_forces_pycuda(r_vectors, *args, **kwargs):
    
    # get number of beads
    number_of_beads = r_vectors.size // 3
    # set the vector force
    forces = np.zeros((number_of_beads, 3))
    
    return forces


def calc_gravitational_forces(r_vectors, bodies_fibers, *args, **kwargs):
    """
    this function computes the gravitational forces.

    Parameters
    ----------
    r_vectors : TYPE
        DESCRIPTION.
    bodies_fibers : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    forces : TYPE
        DESCRIPTION.

    """
    # get number of beads
    number_of_beads = r_vectors.size // 3
    # set the vector force
    forces = np.zeros((number_of_beads, 3))
    
    # get parameters form entries
    W = kwargs.get('weight_per_unit_length')
    
    ind_r = 0
    # loop over fibers
    for k, f in enumerate(bodies_fibers):
        ind_r_1 = ind_r + f.NF
        forces[ind_r:ind_r_1, 2] = -(W*f.fiber_length / f.NF) * np.ones((1, f.NF))
        ind_r += f.NF
         
    return forces


def calc_bending_forces(r_vectors, bodies_fibers, *args, **kwargs):
    """
    Parameters
    ----------
    r_vectors : TYPE
        DESCRIPTION.
    bodies_fibers : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    forces : TYPE
        DESCRIPTION.

    """
    # get number of beads
    number_of_beads = r_vectors.size // 3
    # set the vector force
    forces = np.zeros((number_of_beads, 3))
    
    # get parameters form entries
    W = kwargs.get('weight_per_unit_length')
    B = kwargs.get('elasto_gravitational_number')
    
    ind=0
    for k, f in enumerate(bodies_fibers):
        # get the bead positions vector
        fiber_r_vectors = f.get_fiber_r_vectors()
        # get centerline distance
        DL = f.centerline_distance
        # compute the bending modulus
        KB = W * (f.fiber_length)**3 / B
        
        # first and second beads
        t_1 = fiber_r_vectors[0, :] - fiber_r_vectors[1, :]
        t_2 = fiber_r_vectors[1, :] - fiber_r_vectors[2, :]
        t_3 = fiber_r_vectors[2, :] - fiber_r_vectors[3, :]
        l_1 = np.linalg.norm(t_1)
        l_2 = np.linalg.norm(t_2)
        l_3 = np.linalg.norm(t_3)
        
        forces[ind,:]   = (KB / DL) * (t_2/l_1/l_2 + np.dot(t_1,t_2)*(-t_1)/l_1**3/l_2)
        branch_1        = (t_1 - t_2)/l_1/l_2      + np.dot(t_1, t_2)*(t_1/l_1**3/l_2  - t_2/l_1/l_2**3)
        branch_2        = t_3/l_2/l_3              + np.dot(t_2,t_3)*(-t_2)/l_2**3/l_3    
        forces[ind+1,:] = (KB / DL) * (branch_1 + branch_2)
        
        # third to (N - 2)th beads
        for i in range(2,f.NF - 2):
            t_n_1 = fiber_r_vectors[i-2, :] - fiber_r_vectors[i-1,:]
            t_n_2 = fiber_r_vectors[i-1, :] - fiber_r_vectors[i,  :]
            t_n_3 = fiber_r_vectors[i,   :] - fiber_r_vectors[i+1,:]
            t_n_4 = fiber_r_vectors[i+1, :] - fiber_r_vectors[i+2,:] 
            l_n_1 = np.linalg.norm(t_n_1)
            l_n_2 = np.linalg.norm(t_n_2)
            l_n_3 = np.linalg.norm(t_n_3)
            l_n_4 = np.linalg.norm(t_n_4)
            
            branch_1 = -t_n_1/l_n_1/l_n_2          + np.dot(t_n_1, t_n_2)*t_n_2/l_n_1/l_n_2**3
            branch_2 = (t_n_2 - t_n_3)/l_n_2/l_n_3 + np.dot(t_n_2, t_n_3)*(t_n_2/l_n_2**3/l_n_3 - t_n_3/l_n_2/l_n_3**3)
            branch_3 = t_n_4/l_n_3/l_n_4           + np.dot(t_n_3, t_n_4)*(-t_n_3)/l_n_3**3/l_n_4
            forces[ind+i,:] = (KB / DL) * (branch_1 + branch_2 + branch_3)
            
        # (N - 1)th and Nth  beads
        t_l_1 = fiber_r_vectors[f.NF-4, :] - fiber_r_vectors[f.NF-3, :]
        t_l_2 = fiber_r_vectors[f.NF-3, :] - fiber_r_vectors[f.NF-2, :]
        t_l_3 = fiber_r_vectors[f.NF-2, :] - fiber_r_vectors[f.NF-1, :]
        l_l_1 = np.linalg.norm(t_l_1)
        l_l_2 = np.linalg.norm(t_l_2)
        l_l_3 = np.linalg.norm(t_l_3)
        
        branch_1 = -t_l_1/l_l_1/l_l_2          + np.dot(t_l_1, t_l_2)*t_l_2/l_l_1/l_l_2**3
        branch_2 = (t_l_2 - t_l_3)/l_l_2/l_l_3 + np.dot(t_l_2, t_l_3)*(t_l_2/l_l_2**3/l_l_3 - t_l_3/l_l_2/l_l_3**3)
        forces[ind+f.NF-2, :] = (KB / DL) * (branch_1 + branch_2)
        
        branch_1_last = -t_l_2 /l_l_2/l_l_3
        branch_2_last =                         np.dot(t_l_2, t_l_3 )*t_l_3 /l_l_2/l_l_3**3
        forces[ind+f.NF-1, :] = (KB / DL) * (branch_1_last + branch_2_last)
        
        ind += f.NF
            
    return forces 


def calc_stretching_forces(r_vectors, bodies_fibers, *args, **kwargs):
    """
    Parameters
    ----------
    r_vectors : TYPE
        DESCRIPTION.
    bodies_fibers : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    forces : TYPE
        DESCRIPTION.

    """
    # get number of beads
    number_of_beads = r_vectors.size // 3
    # set the vector force
    forces = np.zeros((number_of_beads, 3))
    
    # get parameters form entries
    W = kwargs.get('weight_per_unit_length')
    B = kwargs.get('elasto_gravitational_number')
    
    ind=0
    for k, f in enumerate(bodies_fibers):
        # get the bead positions vector
        fiber_r_vectors = f.get_fiber_r_vectors()
        # get centerline distance
        DL = f.centerline_distance
        # compute the stretching modulus
        S = (4.0 * W * f.fiber_length**3) / B / (0.5 * DL)**2
        
        for i in range(f.NF):
            if i > 0:
                t_n = fiber_r_vectors[i-1, :] - fiber_r_vectors[i,:]
                l_n = np.linalg.norm(t_n)
                
                branch_1 = - (S / DL) * (l_n - DL) * (-t_n / l_n)
            else:
                branch_1 = 0.
            if i < f.NF - 1:
                t_n_1 = fiber_r_vectors[i, :] - fiber_r_vectors[i+1,:]
                l_n_1 = np.linalg.norm(t_n_1)
                
                branch_2 = - (S / DL) * (l_n_1 - DL) * (t_n_1 / l_n_1)
            else:
                branch_2 = 0.
            
            forces[ind+i, :] = branch_1 + branch_2
    
        ind += f.NF
        
    return forces 

    
    
    
    


        

        
    
    
    
    
    
    
    


     
    
        
    
    
    
    
    
    
        
