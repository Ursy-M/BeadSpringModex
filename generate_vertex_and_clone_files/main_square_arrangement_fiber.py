
"""
Small script file to generate .vertex and .clone files for 4 straight fibers
whose centers coincide with the corners of a horizontal square.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# set entries
a_bead_fiber = 1e-3
dist_factor = 2.2
dist = dist_factor * a_bead_fiber
number_of_beads_per_fiber = 20
length = number_of_beads_per_fiber * dist
offset = 0.268 * length * np.sqrt(2) * 0.5
n_fibers = 4

# define center positions
center_x = [0.0, offset, 0.0, -offset]
center_y = [-offset, 0.0, offset, 0.0]

# set an array for bead coordinates
coordinates = np.zeros((int(n_fibers*number_of_beads_per_fiber), 3), dtype=float)
# set the reference configuration along the z-axis
ref_configuration = np.arange(0, number_of_beads_per_fiber, 1) * dist 
ref_configuration -= np.mean(ref_configuration)

for i in range(n_fibers):
    coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,0] = center_x[i] * np.ones(number_of_beads_per_fiber)
    coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,1] = center_y[i] * np.ones(number_of_beads_per_fiber)
    coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,2] = ref_configuration


# plot
plot_3d = True
# set figure
if plot_3d == True:
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    for i in range(n_fibers):
        ax.plot3D(coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,0],\
                  coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,1], \
                  coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,2],\
                  color='gray')
    
    for i in range(int(n_fibers*number_of_beads_per_fiber)):    
        max_step_theta = 2*np.pi
        max_step_phi = np.pi
        N = 50
        stride=2
        theta = np.linspace(0, max_step_theta, N)
        phi = np.linspace(0, max_step_phi, N)
        
        X0co = coordinates[i,0]  
        Y0co = coordinates[i,1]  
        Z0co = coordinates[i,2]  
        
        Xco = X0co + a_bead_fiber*np.outer(np.cos(theta), np.sin(phi))
        Yco = Y0co + a_bead_fiber*np.outer(np.sin(theta), np.sin(phi))
        Zco = Z0co + a_bead_fiber*np.outer(np.ones(np.size(theta)), np.cos(phi))
    
        ax.plot_surface(Xco, Yco, Zco, linewidth=0.0, cstride=stride, \
                        rstride=stride, color='gray')                    
        
        set_axes_equal(ax)
        ax.set_xlabel(r'$x$', fontsize=20)
        ax.set_ylabel(r'$y$', fontsize=20)
        ax.set_zlabel(r'$z$', fontsize=20)
        
elif plot_3d == False:
    fig, ax = plt.subplots()                                   # in xz plane
    for i in range(n_fibers):
        ax.plot(coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,0], \
                coordinates[i*number_of_beads_per_fiber:(i+1)*number_of_beads_per_fiber,2], color='gray')
    for i in range(int(n_fibers*number_of_beads_per_fiber)):
        max_step = 2*np.pi
        step = np.pi/100
        theta = np.arange(0, max_step, step)
        X0c = coordinates[i,0]  
        Z0c = coordinates[i,2]  
        Xc = X0c + a_bead_fiber*np.cos(theta)
        Zc = Z0c + a_bead_fiber*np.sin(theta)
        
        ax.plot(Xc, Zc, linestyle = '-', color = 'gray')
        ax.set_ylim([-length/2, length/2])
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$', fontsize=20)
        ax.set_ylabel(r'$z$', fontsize=20)
        ax.tick_params(axis = 'x', which='major', direction = 'in', top = True)
        ax.tick_params(axis = 'x', which='minor', direction = 'in', top = True)
        ax.tick_params(axis = 'y', which='major', direction = 'in', right = True)
        ax.tick_params(axis = 'y', which='minor', direction = 'in', right = True)
        fig.tight_layout()

# save 
pos_fiber = coordinates
# generate clones files
filename = "square_fiber_N_" + str(number_of_beads_per_fiber) + "_DL_" + str(dist_factor)
directory = "../configurations"
gg = open(directory + "/" + filename + ".clones", 'w')
print(int(n_fibers * number_of_beads_per_fiber), file=gg)
for i in range(len(coordinates)):
    row = pos_fiber[i,:]
    row = '\t'.join(map(str, row))
    print(row, file=gg)

gg.close()

# generate .vertex files
# first generate a single .vertex file
ff = open(directory + "/" + filename + ".vertex", 'w')
for i in range(number_of_beads_per_fiber):
    ff.write(directory + '/' + 'bead.vertex' + '\n')

ff.close()
plt.show()
