# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:44:46 2020

@author: Ursy
"""

import sys
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


# =============================================================================
# main
# =============================================================================
path = 'output_square_10'                         # TODO : change this entry manually     
current_directory = os.getcwd()
directory = current_directory + '/' + path + '/'


# set some entries
# fiber bead radius 
a = 1e-03                                              
# number of fiber beads                                                    
Nb = 20
# centerline factor
centerline_factor = 2.2
# length                                 
length = Nb * centerline_factor * a
# viscosity
eta = 1.0
# weight per unit length                                                     
W = 1.0
# number of fibers
n_fibers = 4

# setling time
unit_time_gravity = length*eta/W

# load ouput data
y = np.loadtxt(directory + "run.output_positions.csv", delimiter=";")
tk = np.loadtxt(directory + "run.output_times.csv", delimiter=";")
# get number of time step
TOTAL_STEPS = y.shape[0]

plot_steps = 50
plot_now = plot_steps - 1 
fig, ax = plt.subplots(figsize=(5, 15))

for k in range(1, TOTAL_STEPS):
    plot_now = plot_now + 1
    
    all_x = list()
    all_y = list()
    all_z = list()
    
    Ny = int(y.shape[1]/3)
    if (k==0):
        for kk in range(Ny) :
            all_x.append(y[:,3*kk  ][0])
            all_y.append(y[:,3*kk+1][0])
            all_z.append(y[:,3*kk+2][0])
            
        all_x[:] = [x/length for x in all_x]
        all_y[:] = [y/length for y in all_y]
        all_z[:] = [z/length for z in all_z]
        
    else :
        for kk in range(Ny) :
            all_x.append(y[:,3*kk  ][k])
            all_y.append(y[:,3*kk+1][k])
            all_z.append(y[:,3*kk+2][k])
            
        all_x[:] = [x/length for x in all_x]
        all_y[:] = [y/length for y in all_y]
        all_z[:] = [z/length for z in all_z]
        
    if (plot_now == plot_steps):
        print('t/T=', int(tk[k]/unit_time_gravity))
        
        for i in range(n_fibers):
            ax.plot((all_x[i*Nb:(i+1)*Nb]),(all_z[i*Nb:(i+1)*Nb]),'-', color='gray')
            
        # plot spheres for fiber and set figure parameters
        for ii in range(Ny):
            max_step = 2*np.pi
            step = np.pi/100
            theta = np.arange(0, max_step, step)
            X0c = all_x[ii] 
            Z0c = all_z[ii] 
            Xc = X0c + (a/length)*np.cos(theta)
            Zc = Z0c + (a/length)*np.sin(theta)
            
            ax.plot(Xc, Zc, linestyle ='-', color = 'gray')
            ax.set_aspect('equal')
        
        ax.set_xlim(-4.0 , 4.0)
        ax.set_ylim(-80, 0.5)
        ax.set_xticks([-4.0, 0, 4.0])
        #ax.set_yticks([-Lim, 0, Lim])
        ax.set_xlabel(r'$x/L$', fontsize=20)
        ax.set_ylabel(r'$z/L$', fontsize=20)
        ax.tick_params(axis = 'x', which='major', direction = 'in', top = True)
        ax.tick_params(axis = 'x', which='minor', direction = 'in', top = True)
        ax.tick_params(axis = 'y', which='major', direction = 'in', right = True)
        ax.tick_params(axis = 'y', which='minor', direction = 'in', right = True)
        fig.tight_layout()
    
        
        plt.draw()
        plt.pause(0.02)
        
    if (plot_now == plot_steps):
        plot_now = 0
        
