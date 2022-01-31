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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter

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

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg';


# =============================================================================
# define some useful functions
# =============================================================================
def set_axes_equal(ax):
    
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# =============================================================================
# main
# =============================================================================
path = 'output'     
current_directory = os.getcwd()
directory = current_directory + '/' + path + '/'


# set some entries
# fiber bead radius 
a = 1e-03                                              
# number of fiber beads                                                    
Nb = 20
# length                                 
LS = Nb*2.2*a
# viscosity
eta = 1.0
# weight per unit length                                                     
W = 1.0
# number of fibers
n_fibers = 2

# setling time
unit_time_gravity = LS*eta/W

# load ouput data
y = np.loadtxt(directory + "run.output_positions.csv", delimiter=";")
tk = np.loadtxt(directory + "run.output_times.csv", delimiter=";")
# get number of time step
TOTAL_STEPS = y.shape[0]
 
plot_3d = False
# plot setup
plot_steps = 30                                            
plot_now = plot_steps - 1
if plot_3d==False:
    fig, ax = plt.subplots(figsize=(7, 7))
elif plot_3d==True:
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
plt.subplots_adjust(bottom=0.25)


# Movie setup
np.random.seed(2984)
metadata=dict(title="Sedimentation",artist="Low_Reynolds_Number",comment="p");
writer=FFMpegWriter(fps=2,metadata=metadata)


Lim = 0.5


# build the movie 
with writer.saving(fig, directory + "Movie.mov",100):
    for k in range(TOTAL_STEPS - 1):
        plot_now = plot_now + 1
        
        all_x = list()
        all_y = list()
        all_z = list()
        
        Ny = int(y.shape[1]/3)
        if (k==0):
            for kk in range(Ny) :
                all_x.append(y[:,(3*kk)][0])
                all_y.append(y[:,(3*kk+1)][0])
                all_z.append(y[:,3*kk+2][0])
                
            all_x[:] = [x/LS for x in all_x]
            all_y[:] = [y/LS for y in all_y]
            all_z[:] = [z/LS for z in all_z]
            x_m = np.mean(all_x)
            y_m = np.mean(all_y)
            z_m = np.mean(all_z)
            
        else :
            for kk in range(Ny) :
                all_x.append(y[:,(3*kk)][k])
                all_y.append(y[:,(3*kk+1)][k])
                all_z.append(y[:,3*kk+2][k])
                
            all_x[:] = [x/LS for x in all_x]
            all_y[:] = [y/LS for y in all_y]
            all_z[:] = [z/LS for z in all_z]
            x_m = np.mean(all_x)
            y_m = np.mean(all_y)
            z_m = np.mean(all_z)
            
        if (plot_now == plot_steps):
            if plot_3d==False:
                if n_fibers==1:
                    ax.plot((all_x - x_m),(all_z - z_m),'-')
                elif n_fibers==2:
                    ax.plot((all_x[:Nb] - x_m),(all_z[:Nb] - z_m),'-', color='gray')
                    ax.plot((all_x[Nb:] - x_m),(all_z[Nb:] - z_m),'-', color='gray')
            elif plot_3d==True:
                if n_fibers==1:
                    ax.plot3D((all_x - x_m), (all_y - y_m), (all_z - z_m),color='gray', linestyle='-')
                elif n_fibers==2:
                    ax.plot3D((all_x[:Nb] - x_m), (all_y[:Nb] - y_m), (all_z[:Nb] - z_m),color='gray', linestyle='-')
                    ax.plot3D((all_x[Nb:] - x_m), (all_y[Nb:] - y_m), (all_z[Nb:] - z_m),color='gray', linestyle='-')
                
            # plot spheres for fiber and set figure parameters
            if plot_3d==False:
                for ii in range(Ny):
                    max_step = 2*np.pi
                    step = np.pi/100
                    theta = np.arange(0, max_step, step)
                    X0c = all_x[ii] - x_m
                    Z0c = all_z[ii] - z_m
                    Xc = X0c + (a/LS)*np.cos(theta)
                    Zc = Z0c + (a/LS)*np.sin(theta)
                    
                    ax.plot(Xc, Zc, linestyle ='-', color = 'gray')
                    ax.set_aspect('equal')
                
                ax.set_xlim(-Lim , Lim)
                ax.set_ylim(-Lim, Lim)
                ax.set_xticks([-Lim, 0, Lim])
                ax.set_yticks([-Lim, 0, Lim])
                ax.set_xlabel(r'$(x - x_{mean})/L$', fontsize=20)
                ax.set_ylabel(r'$(z - z_{mean})/L$', fontsize=20)
                ax.tick_params(axis = 'x', which='major', direction = 'in', top = True)
                ax.tick_params(axis = 'x', which='minor', direction = 'in', top = True)
                ax.tick_params(axis = 'y', which='major', direction = 'in', right = True)
                ax.tick_params(axis = 'y', which='minor', direction = 'in', right = True)
                ax.set_title(r'$t/T= $' + '\t' + str(int(tk[k]/unit_time_gravity)), fontsize=20)
            elif plot_3d==True:
                for ii in range(Ny):
                    max_step_theta = 2*np.pi
                    max_step_phi = np.pi
                    N = 50
                    stride=2
                    theta = np.linspace(0, max_step_theta, N)
                    phi = np.linspace(0, max_step_phi, N)
                    
                    X0co = all_x[ii] - x_m
                    Y0co = all_y[ii] - y_m
                    Z0co = all_z[ii] - z_m
                    
                    Xco = X0co + (a/LS)*np.outer(np.cos(theta), np.sin(phi))
                    Yco = Y0co + (a/LS)*np.outer(np.sin(theta), np.sin(phi))
                    Zco = Z0co + (a/LS)*np.outer(np.ones(np.size(theta)), np.cos(phi))
                
                    ax.plot_surface(Xco, Yco, Zco, linewidth=0.0, cstride=stride, \
                                    rstride=stride, color='gray')                        
                    set_axes_equal(ax)
                
                ax.set_xlim(-2*Lim , 2*Lim)
                ax.set_ylim(-2*Lim, 2*Lim)
                ax.set_zlim(-2*Lim, 2*Lim)
                ax.set_xticks([-2*Lim, 0, 2*Lim])
                ax.set_xticks([-2*Lim, 0, 2*Lim])
                ax.set_zticks([-2*Lim, 0, 2*Lim])
                ax.set_xlabel(r'$(x - x_{mean})/L$', fontsize=10)
                ax.set_ylabel(r'$(y - y_{mean})/L$', fontsize=10)
                ax.set_zlabel(r'$(z - z_{mean})/L$', fontsize=10)
                
            
            plt.draw()
            plt.pause(0.02)
            

            writer.grab_frame()
            plt.cla()
            
        
        if (plot_now == plot_steps):
            plot_now = 0
            
