
"""
Small script  file to generate .vertex and .clone files for a random suspension of fibers
in a quasi 2D square lattice.
"""

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quaternion.quaternion import Quaternion

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
# set entries
a_bead_fiber = 1e-3                                     # radius bead fiber 
dist_factor = 2.2
dist = dist_factor * a_bead_fiber
number_of_beads_per_fiber = 20
length = number_of_beads_per_fiber * dist
N_x = 10                                                  # number of fibers along x-axis 
N_z = 10                                                  # number of fibers along z-axis 
L_x = 20 * length                                        # length along x-axis 
L_z = 20 * length                                        # length along z-axis 
depth   = 0                                              # average depth
rand_factor = 0


# define the reference configuration (straight fiber oriented horizontaly)
ref_configuration = np.zeros((number_of_beads_per_fiber, 3))
ref_configuration[0:number_of_beads_per_fiber,0]= np.arange(0, number_of_beads_per_fiber, 1) * dist

# set an array for fiber positions
position = []

dx = L_x / N_x
dz = L_z / N_z
dy = depth 

for i in range(N_z):
    for j in range(N_x):
        x = float(j) *  dx +  (2.0 * np.random.rand() - 1.0) * rand_factor
        z = float(i) *  dz +  (2.0 * np.random.rand() - 1.0) * rand_factor
        y = dy 
        position.append([x, y, z])
        
position = np.array(position)

# set the number of fibers
print('number of fibers: ', N_x * N_z)
n_fibers = len(position)

# generate random orientations
orientation = np.zeros((n_fibers, 4))
for k in range(n_fibers):
    factor = np.random.rand()
    s = np.array([np.cos(factor*np.pi/2)])
    p = np.sin(factor*np.pi/2) * np.array([0., factor*np.pi, 0.]) / np.linalg.norm(np.array([0., factor*np.pi, 0.]))
    q = np.concatenate([s, p])
    orientation[k] = q
    
orientations = []
for k in range(len(orientation)):
    orientations.append(Quaternion(orientation[k]))

# compute the bead coordinates
coordinates = []
for k in range(len(position)):
    rotation_matrix = orientations[k].rotation_matrix()
    r_vectors = np.dot(ref_configuration, rotation_matrix.T)
    r_vectors += position[k]
    coordinates.append(r_vectors)
    
coordinates = np.array(coordinates)

        
# plot
fig, ax = plt.subplots()                                   # in xz plane
for i in range(n_fibers):
    ax.plot(coordinates[i][:,0], coordinates[i][:,2], color='gray')
    for j in range(number_of_beads_per_fiber):
        max_step = 2*np.pi
        step = np.pi/100
        theta = np.arange(0, max_step, step)
        X0c = coordinates[i][j,0]  
        Z0c = coordinates[i][j,2]  
        Xc = X0c + a_bead_fiber*np.cos(theta)
        Zc = Z0c + a_bead_fiber*np.sin(theta)
        
        ax.plot(Xc, Zc, linestyle = '-', color = 'gray')
        ax.set_aspect('equal')
    
    
ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$z$', fontsize=20)
ax.tick_params(axis = 'x', which='major', direction = 'in', top = True)
ax.tick_params(axis = 'x', which='minor', direction = 'in', top = True)
ax.tick_params(axis = 'y', which='major', direction = 'in', right = True)
ax.tick_params(axis = 'y', which='minor', direction = 'in', right = True)
#ax.set_xlim([0, L_x])
#ax.set_ylim([0, L_z])
#ax.set_xticks([0, L_x])
#ax.set_yticks([0, L_z])
fig.tight_layout()


# save 
# generate clones files
filename = "straight_fiber_N_" + str(number_of_beads_per_fiber) + "_DL_" + str(dist_factor)
directory = "../configurations"
gg = open(directory + "/" + filename + ".clones", 'w')
print(int(n_fibers * number_of_beads_per_fiber), file=gg)

for i in range(n_fibers):    
    for j in range(number_of_beads_per_fiber):
        row = coordinates[i][j,:]
        row = '\t'.join(map(str, row))
        print(row, file=gg)

gg.close()

# generate .vertex files
# first generate a single .vertex file
ff = open(directory + "/" + filename + ".vertex", 'w')
for i in range(number_of_beads_per_fiber):
    ff.write(directory + '/' + 'bead.vertex' + '\n')

ff.close()    

   