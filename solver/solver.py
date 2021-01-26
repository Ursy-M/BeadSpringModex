# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:22:15 2020

@author: Ursy
"""

import numpy as np
from scipy.integrate import ode


def BDF_method(RPY, initial_values, t, dt, V_fiber, Nbeads, V, args):
    
    """ ODE solver interface : integrate.ode class """
    """ see http://www.netlib.org/ode to learn more about ode solver """
    """
    input : 
        - RPY : Rotne Prager Yamakawa's tensor to provide the right hand side
        - initial_values : initial conditions
        - t : an array of simulation time
        - dt : step size
        - V_fiber : an array contains all fiber bead velocities at each time step
        - Nbeads : number of beads
        - V : an array of Nx3 contains bead velocities at given time step
        - args : arguments
    """
    """ return y (bead positions), V_fiber, all_t (simulation time)"""
    
    r = ode(RPY)
    r.set_integrator('vode', method = 'BDF', with_jacobian = True);
    r.set_initial_value(initial_values, t.min());
    r.set_f_params(args);
    
    y = np.zeros((len(t),len(initial_values)))
    all_t = np.zeros((len(t) ,1), dtype=float)
    idx = 0
    
    while r.successful() and r.t < t[-1] :
        y[idx, :] = r.y
        V_fiber[idx,:] = np.squeeze(np.asarray(V.reshape(1,3*Nbeads)))
        all_t[idx,:] = t[idx]
        r.integrate(r.t + dt)
        idx += 1
    
    return y, V_fiber, all_t