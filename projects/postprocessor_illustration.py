# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:44:46 2020

@author: Ursy
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)


#plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg';
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe';

# load input data
f = np.loadtxt("inputdata.csv", skiprows=1, delimiter = ";", dtype=str)
     
current_directory = os.getcwd()
path = 'outputdata'
directory = current_directory + '/' + path + '/'

# parameters
# fiber bead radius 
a = float(f[0])                                              
# number of beads of the fiber                                                    
Nb = int(f[1])
# length                                 
LS = Nb * 2 * a
# viscosity
mu = float(f[2])
# weight per unit length                                                     
W = float(f[3])
# setling time
unit_time_gravity = LS * mu / W

# define the domain
Lim = 0.5

# load ouput data
y = np.loadtxt(directory + "outputpositions.csv", delimiter=";")
tk = np.loadtxt(directory + "outputtime.csv", delimiter=";")

# get the number of time step
TOTAL_STEPS = y.shape[0]
 
# plot setup
plot_steps = 30                                            
plot_now = plot_steps - 1
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.25)

# Movie setup
np.random.seed(2984)
metadata=dict(title="Sedimentation",artist="Low_Reynolds_Number",comment="p");
writer=FFMpegWriter(fps=2,metadata=metadata)


# build the movie 
with writer.saving(fig, directory + "Movie.mp4", 100):
    for k in range(TOTAL_STEPS - 1):
        plot_now = plot_now + 1
        
        all_x = list()
        all_y = list()
        
        all_f_x = list()
        all_f_y = list()
        
        Ny = int(y.shape[1]/3)
        if (k==0):
            for kk in range(Ny) :
                all_x.append(y[:,(3 * kk)][0])
                all_y.append(y[:,3 * kk+1][0])
                
            all_x[:] = [x / LS for x in all_x]
            all_y[:] = [y / LS for y in all_y]
            x_m = np.mean(all_x)
            y_m = np.mean(all_y)
            
        else :
            for kk in range(Ny) :
                all_x.append(y[:,(3 * kk)][k])
                all_y.append(y[:,3 * kk+1][k])
                
            all_x[:] = [x / LS for x in all_x]
            all_y[:] = [y / LS for y in all_y]
            x_m = np.mean(all_x)
            y_m = np.mean(all_y)
            
        if (plot_now == plot_steps) :  
            
            ax.plot((all_x - x_m),(all_y - y_m),'-')
            
            # plot spheres --> fiber
            for ii in range(Ny):
                max_step = 2 * np.pi
                step = np.pi / 100
                theta = np.arange(0, max_step, step)
                X0c = all_x[ii] - x_m
                Y0c = all_y[ii] - y_m
                Xc = X0c + (a / LS)* np.cos(theta)
                Yc = Y0c + (a / LS) * np.sin(theta)
                
                ax.plot(Xc, Yc, linestyle ='-', color = 'gray')
                ax.set_aspect('equal')
            
            ax.set_xlim(-Lim , Lim)
            ax.set_ylim(-Lim, Lim)
            ax.set_xticks([-Lim, 0, Lim])
            ax.set_yticks([-Lim, 0, Lim])
            ax.set_xlabel(r'$(x - x_{mean})/L$', fontsize=20)
            ax.set_ylabel(r'$(y - y_{mean})/L$', fontsize=20)
            ax.set_title(r'$t/T=$' + str(int(tk[k]/unit_time_gravity)), fontsize=20)
            
            plt.draw()
            plt.pause(0.02)
            
            
            writer.grab_frame()
            
            plt.cla()
            
        if (plot_now == plot_steps):
            plot_now = 0
            