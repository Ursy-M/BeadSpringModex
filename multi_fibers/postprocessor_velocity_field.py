# author : ursy

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FFMpegWriter
from datetime import datetime

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

# set directories
path = 'output'     
current_directory = os.getcwd()
directory = current_directory + '/' + path + '/'
directory_grid_velocity = directory + 'output_grid_velocity' + '/'

# load output data
grid_coordinates = np.loadtxt(directory_grid_velocity + 'run.grid_coordinates.csv', delimiter=';')
y = np.loadtxt(directory + "run.output_positions.csv", delimiter=';')
tk = np.loadtxt(directory + "run.output_times.csv", delimiter=';')

# set grid data 
n_x = 1000
n_z = 1000
x = grid_coordinates[:,0].reshape(n_x, n_z)
#y = grid_coordinates[:,1].reshape(1000, 1000)
z = grid_coordinates[:,2].reshape(n_x, n_z)


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

# setling time
unit_time_gravity = LS*eta/W

# get the number of steps
n_steps = y.shape[0]


# plot setup
plot_steps = 20                                            
plot_now = plot_steps - 1
Lim = 0.5

fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.25)

# Movie setup
np.random.seed(2984)
metadata=dict(title="Sedimentation",artist="Low_Reynolds_Number",comment="p");
writer=FFMpegWriter(fps=2,metadata=metadata)

start_time = datetime.now()
# build the movie 
with writer.saving(fig, directory + "Movie_velocity_field.mov",100):
    for k in range(1,n_steps - 1):
        plot_now = plot_now + 1
        if k == 1:
            print('building movie...')
        
        all_x = list()
        all_y = list()
        all_z = list()
        
        Ny = int(y.shape[1]/3)
        
        for kk in range(Ny) :
            all_x.append(y[:,(3*kk)][k])
            all_y.append(y[:,(3*kk+1)][k])
            all_z.append(y[:,3*kk+2][k])
    
        x_m = np.mean(all_x)
        y_m = np.mean(all_y)
        z_m = np.mean(all_z)
        
        if (plot_now == plot_steps):
            grid_velocity = np.loadtxt(directory_grid_velocity + 'run.velocity_field_' + str(k) +'.csv', delimiter=';')
            grid_velocity = grid_velocity.reshape(grid_velocity.size // 3, 3)
            u = grid_velocity[:,0]
            v = grid_velocity[:,1]
            w = grid_velocity[:,2]
            
            u = u.reshape(n_x, n_z)
            v = v.reshape(n_x, n_z)
            w = w.reshape(n_x, n_z)
            
            
            #ax.plot((all_x - x_m)/LS, (all_z - z_m)/LS, linestyle='-', color='black', linewidth=10)
            #ax.streamplot((x - x_m)/LS, (z - z_m)/LS, u, w, color='white', density=5, linewidth=0.5,\
            #              integration_direction='forward', arrowstyle='fancy') 
            spd = np.hypot(u, w)
            p = ax.pcolormesh((x - x_m)/LS, (z - z_m)/LS , np.log(spd), shading = 'nearest', cmap=cm.jet)
            
            
            
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
            #fig.tight_layout()
            
            plt.draw()
            plt.pause(0.02)
            
    
            writer.grab_frame()
            plt.cla()
            
        
        if (plot_now == plot_steps):
            plot_now = 0
            
duration = datetime.now() - start_time
print('time :\t' + 'execution\t' + str(duration))
print('\n\n\n# End')
