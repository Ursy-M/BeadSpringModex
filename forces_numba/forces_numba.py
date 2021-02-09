#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:44:59 2020

@author: Ursy
"""

""" external forces applied on fiber and obstacle beads with numba """

import numpy as np
import scipy 


# try to import numba
try :
    from numba import njit, prange
except ImportError:
    print('numba no found')


def stretching_forces_numba(X, a, coef_stretching, N_bead_obs):
    
    """ from Erik Gauger and Holder Stark's paper """
    

    """ inputs :
        - X : array of size Nx3 contains all  bead positions
        - a : fiber bead radius
        - coef_strectching : strecthing modulus E*\pi*a^2 (E being the Young's modulus)
        - N_bead_obs : number of obstacle beads
    """
    
    """ return an array F of size Nx3 """
    
    # get number of beads
    N = X.shape[0]
    # get number of fiber beads
    Nb = N - N_bead_obs
    # get the stretching modulus
    xi_0 = coef_stretching
    # define the vector force
    F = np.zeros((N, 3), dtype = float)
    
    # compute the forces
    for ii in range(Nb):
        if ii>0:
            # compute the variation of postion between beads i - 1 and i
            rim_i_vec =   X[ii - 1, :] - X[ii, :]
            # compute the norm of vector postion x_{i - 1} - x_{i}
            rim_i = scipy.spatial.distance.euclidean(X[ii - 1, :], X[ii, :])
            #compute strecthing force (contribution of bead i - 1 on bead i)
            F_S_1 = -xi_0 * (rim_i - 2*a)*(-rim_i_vec/rim_i) 
        else  :
            F_S_1 = 0 
        if ii < Nb-1:
            # compute the variation of postion between beads i and i + 1
            ri_ip_vec =  X[ii, :] -  X[ii + 1, :]
            # compute the norm of vector postion x_{i} - x_{i + 1}
            ri_ip = scipy.spatial.distance.euclidean(X[ii,:], X[ii+1,:])
            # compute strecthing force (contribution of bead i on bead i + 1)
            F_S_2 =  -xi_0 * (ri_ip - 2*a)*(ri_ip_vec/ri_ip)
        else :
            F_S_2 = 0
        #sum both contributions
        F[ii, :] = F_S_1 + F_S_2
    
    return  F

@njit(parallel = True, fastmath = True)
def bending_forces_numba(X, coef_bending, N_bead_obs):
    
    """ from Erik Gauger and Holder Stark's paper """
    

    """ inputs :
        - X : array of size Nx3 contains all bead positions
        - a : fiber bead radius
        - coef_bending : bending modulus E*\pi*a^4/4 (E being the Young's modulus)
        - N_bead_obs : number of obstacle beads
    """
    
    """ return an array F of size Nx3 """
    
    # get number of beads
    N = X.shape[0]
    # get number of fiber beads
    Nb = N - N_bead_obs
    # define the vector force
    F = np.zeros((N, 3))
    
    # compute bending forces for the first fiber bead
    # compute the tangential vectors
    t1_vec = X[0, :] - X[1, :]
    t2_vec = X[1, :] - X[2, :]
    # compute the norms of the tangential vectors
    l1 = np.linalg.norm(t1_vec)
    l2 = np.linalg.norm(t2_vec)
    # cmpoute force applied on first fiber bead
    F[0, :] = coef_bending * (t2_vec/l1/l2 + np.dot(t1_vec,t2_vec)*(-t1_vec)/l1**3/l2)
    
    # compute bending forces for the second fiber bead
    # compute the tangential vectors
    t1_vec = X[0, :] - X[1, :]
    t2_vec = X[1, :] - X[2, :]
    t3_vec = X[2, :] - X[3, :]
    # compute the norms of the tangential vectors
    l1 = np.linalg.norm(t1_vec)
    l2 = np.linalg.norm(t2_vec)
    l3 = np.linalg.norm(t3_vec)
    # compute force applied on the second fiber bead
    branch_1 = (t1_vec - t2_vec)/l1/l2 + np.dot(t1_vec, t2_vec)*(t1_vec/l1**3/l2  - t2_vec/l1/l2**3)
    branch_2 = t3_vec/l2/l3 + np.dot(t2_vec,t3_vec)*(-t2_vec)/l2**3/l3
    F[1, :] = coef_bending * (branch_1 + branch_2)
    
    # from the third fiber bead to (Nb - 2)th fiber bead
    for ii in prange(2, Nb - 2):
        # compute the tangential vectors
        ti_vec = X[ii - 2, :] - X[ii - 1, :]
        ti_p_1_vec = X[ii - 1, :] - X[ii, :]
        ti_p_2_vec = X[ii, :] - X[ii + 1, :]
        ti_p_3_vec = X[ii + 1, :] - X[ii + 2, :]
        # compute the norms of the tangential vectors
        li = np.linalg.norm(ti_vec)
        li_p_1 = np.linalg.norm(ti_p_1_vec)
        li_p_2 = np.linalg.norm(ti_p_2_vec)
        li_p_3 = np.linalg.norm(ti_p_3_vec)
        # compute the forces applied on the fiber beads from third to (Nb - 2)th
        branch_1 = -ti_vec/li/li_p_1 + np.dot(ti_vec,ti_p_1_vec)*ti_p_1_vec/li/li_p_1**3
        branch_2 = (ti_p_1_vec - ti_p_2_vec)/li_p_1/li_p_2 + np.dot(ti_p_1_vec,ti_p_2_vec)*(ti_p_1_vec/li_p_1**3/li_p_2 - ti_p_2_vec/li_p_1/li_p_2**3)
        branch_3 = ti_p_3_vec/li_p_2/li_p_3 + np.dot(ti_p_2_vec,ti_p_3_vec)*(-ti_p_2_vec)/li_p_2**3/li_p_3
        F[ii, :] = coef_bending * (branch_1 + branch_2 + branch_3)
        
    # compute bending force for the (Nb - 1)th fiber bead
    # compute the tangential vectors
    tn_vec = X[Nb - 4, :] - X[Nb -3 , :]
    tn_p_1_vec = X[Nb - 3, :] - X[Nb - 2, :]
    tn_p_2_vec = X[Nb - 2, :] - X[Nb - 1, :]
    # compute the norms of the tangential vectors
    ln = np.linalg.norm(tn_vec)
    ln_p_1 = np.linalg.norm(tn_p_1_vec)
    ln_p_2 = np.linalg.norm(tn_p_2_vec)
    # compute the force applied on the (Nb - 1)th fiber bead
    branch_1 = -tn_vec/ln/ln_p_1 + np.dot(tn_vec,tn_p_1_vec)*tn_p_1_vec/ln/ln_p_1**3
    branch_2 = (tn_p_1_vec - tn_p_2_vec)/ln_p_1/ln_p_2 + np.dot(tn_p_1_vec,tn_p_2_vec)*(tn_p_1_vec/ln_p_1**3/ln_p_2 - tn_p_2_vec/ln_p_1/ln_p_2**3)
    F[Nb - 2, :] =  coef_bending * (branch_1 + branch_2)
          
    # compute bending force for the last fiber bead
    # compute the tangential vectors
    tn_p_1_vec = X[Nb - 3, :] - X[Nb - 2, :]
    tn_p_2_vec = X[Nb - 2, :] - X[Nb - 1, :]
    # compute the norms of the tangential vectors
    ln_p_1 = np.linalg.norm(tn_p_1_vec)
    ln_p_2 = np.linalg.norm(tn_p_2_vec)
    # compute the force applied on the last fiber bead
    branch_1 = -tn_p_1_vec/ln_p_1/ln_p_2
    branch_2 = np.dot(tn_p_1_vec,tn_p_2_vec)*tn_p_2_vec/ln_p_1/ln_p_2**3
    F[Nb - 1, :] = coef_bending * (branch_1 + branch_2)
        
    return F
