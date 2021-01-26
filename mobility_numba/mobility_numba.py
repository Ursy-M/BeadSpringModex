# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:23:01 2020

@author: ursy
"""

""" Mobilities in unbounded media """


import numpy as np

# try to import numba
try:
    from numba import njit, prange
except ImportError:
    print('numba no found')
    

def mobility_matrix_same_bead_size(X, mu, a) :
    
    """ from P. J. Zuk, E. Wajnryb, K. A. Mizerski and P. Szymczak's paper """
    
    """ inputs :
        - X : array of size Nx3 contains all  bead positions
        - mu : dynamic viscosity
        - a : bead radius
    """
    
    """ return a matrix of size 3Nx3N """
    
    # get the inverse of the drag coefficient
    factor = 1.0/(6.0*np.pi*mu)
    #extract variables
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    #compute distance between beads
    dx = x - x[:, None]
    dy = y - y[:, None]
    dz = z - z[:, None]
    # compute the length
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    #compute scalar functions f(r) and g(r)
    fr = np.zeros_like(dr)
    gr = np.zeros_like(dr)
    sel = dr != 0.
    sel_zero = dr == 0.
    
    fr[sel] = factor * (0.75 / dr[sel] + a**2 / (2.0 * dr[sel]**3))
    gr[sel] = factor * (0.75 / dr[sel]**3 - 1.5 * a**2 / dr[sel]**5)
    
    fr[sel_zero] = (factor/a)
    
    # buid the mobility matrix M_beads
    M_beads = np.zeros((X.size, X.size), dtype = float)
    
    M_beads[0::3, 0::3] = fr + gr*dx*dx
    M_beads[0::3, 1::3] =      gr*dx*dy
    M_beads[0::3, 2::3] =      gr*dx*dz
    
    M_beads[1::3, 0::3] =      gr*dy*dx
    M_beads[1::3, 1::3] = fr + gr*dy*dy
    M_beads[1::3, 2::3] =      gr*dy*dz
    
    M_beads[2::3, 0::3] =      gr*dz*dx
    M_beads[2::3, 1::3] =      gr*dz*dy
    M_beads[2::3, 2::3] = fr + gr*dz*dz
    
    return M_beads


@njit(parallel = True, fasmath = True)
def mobility_matrix_different_bead_size_loop(X, mu, radi_vec) :
    
    """ from P. J. Zuk, E. Wajnryb, K. A. Mizerski and P. Szymczak's paper """
    
    """ inputs :
        - X : array of size Nx3 contains all  bead positions
        - mu : dynamic viscosity
        - radi_vec : array of size Nx1 contains all bead radius
    """
    
    """ return a matrix of size 3Nx3N """
    
    # get the number of beads
    Nbeads = X.shape[0]
    # compute the inverse of the drag coefficient
    factor = 1.0/(6.0*np.pi*mu)
    
    # buid the mobility matrix M_beads
    M_beads = np.array([np.zeros(3*Nbeads) for _ in range(3*Nbeads)])
    for j in prange(Nbeads):
        for k in prange(Nbeads):
            if j != k :
                # compute the position vector between bead j and bead k
                Rjk_vec = X[j,:] - X[k,:]
                # compute the norm of the position vector
                Rjk = np.linalg.norm(Rjk_vec)
                # compute the prefactors of the RPY tensor 
                xi_0 = 1/(8*np.pi*mu*Rjk)
                xi_1 = 1 + (radi_vec[j]**2 + radi_vec[k]**2)/(3*Rjk**2)
                xi_2 = 1 - (radi_vec[j]**2 + radi_vec[k]**2)/(Rjk**2)
                
                M_beads[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = (xi_0*(xi_1*np.identity(3) \
                               + xi_2*np.outer(Rjk_vec, Rjk_vec)/(np.maximum(Rjk, np.finfo(float).eps)**2)))
    
            elif j == k :
                M_beads[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = (factor/radi_vec[j])*np.identity(3)

    return M_beads   




    
    