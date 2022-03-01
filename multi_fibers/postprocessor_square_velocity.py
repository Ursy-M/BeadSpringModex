# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:44:46 2020

@author: Ursy
"""

import numpy as np
import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from matplotlib import rc
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
except ImportError:
    print('You don’t have a working LaTeX installation on your computer or the required files are not in your system environment variables.')
   
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)

# set directory 
folder_name = 'output_square_'     
current_directory = os.getcwd()
# define list of be
list_be = [10, 100, 1000]

# set linestyles and colors
linestyle_str = ['solid', 'dotted', 'dashdot']
color_str     = ['black', 'red', 'blue']

# set some entries
# fiber bead radius 
a = 1e-03                                              
# number of fiber beads                                                    
number_of_beads_per_fiber = 20
# centerline factor
centerline_factor = 2.2
# length                                 
length = number_of_beads_per_fiber * centerline_factor * a
# viscosity
eta = 1.0
# weight per unit length                                                     
W = 1.0
# number of fibers
n_fibers = 4
# aspect ratio
kappa_inv = length / (2 * a)

# setling time
unit_time_gravity = length*eta/W

# compute theoretically U_perp
factor = 1. / (2 * np.pi * eta)
U_perp = 0.5 * W * (np.log(4*kappa_inv) - 0.5) * factor
 
fig, ax = plt.subplots()

# lopp over the values of be 
for ind, Be in enumerate(list_be):
    path = folder_name + str(Be)
    directory = current_directory + '/' + path + '/' 
    sol_velocities = np.loadtxt(directory + "run.output_velocities.csv", delimiter=";")     
    tk = np.loadtxt(directory + "run.output_times.csv", delimiter=";")
    tk[:] = [x/unit_time_gravity for x in tk]
    # get number of steps
    TOTAL_STEPS = sol_velocities.shape[0]
    # total number of beads     
    N = int(sol_velocities.shape[1]/3)
    
    # set a list to contain the velocity of the center of mass at each time steps 
    all_vel_normalized = []
    
    for k in range(TOTAL_STEPS - 1):
        all_v_over_fibers = []
        all_vx            = []
        all_vy            = []
        all_vz            = []
    
        for i in range(N):
            all_vx.append(sol_velocities[:, 3*i    ][k])
            all_vy.append(sol_velocities[:, 3*i + 1][k])
            all_vz.append(sol_velocities[:, 3*i + 2][k])
        
        for fiber in range(n_fibers):    
            vx_m = np.mean(all_vx[fiber*number_of_beads_per_fiber:(fiber+1)*number_of_beads_per_fiber])
            vy_m = np.mean(all_vy[fiber*number_of_beads_per_fiber:(fiber+1)*number_of_beads_per_fiber])
            vz_m = np.mean(all_vz[fiber*number_of_beads_per_fiber:(fiber+1)*number_of_beads_per_fiber])
            # compute the norm    
            v_cm_norm = np.linalg.norm(np.array([vx_m, vy_m, vz_m]))
            
            # add to list
            all_v_over_fibers.append(v_cm_norm)
        
        # compute the mean and add it to the general list
        v_mean = np.mean(np.array(all_v_over_fibers))
        all_vel_normalized.append(v_mean/U_perp)
    
    ax.plot(tk[0:TOTAL_STEPS-1], all_vel_normalized, linewidth=2.5, \
            linestyle=linestyle_str[ind], color=color_str[ind],\
                label=r'$Be$' + "\t" + r'$=$' + "\t" + r'${arg}$'.format(arg=str(Be)))
        
            
number_of_set_time = 150            
ax.set_xlim(0 , number_of_set_time)
ax.set_ylim(0, 3)
ax.set_xlabel(r'$t/T$', fontsize=20)
ax.set_ylabel(r'$\overline{U}/U_{\perp}$', fontsize=20)
ax.tick_params(axis = 'x', which='major', direction = 'in', top = True)
ax.tick_params(axis = 'x', which='minor', direction = 'in', top = True)
ax.tick_params(axis = 'y', which='major', direction = 'in', right = True)
ax.tick_params(axis = 'y', which='minor', direction = 'in', right = True)
ax.legend(loc=4)
fig.tight_layout()
plt.subplots_adjust(left=0.15)        
        
