# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:53:07 2020

@author: Ursy
"""


import sys
import numpy as np
import os
import shutil
from datetime import datetime

sys.path.append('.')
sys.path.append('../')

from mobility_numba import mobility_numba as mob
from forces_numba import forces_numba as fo
from solver import solver as sv

startTime = datetime.now()

# load  input file
f = np.loadtxt("inputdata.csv", skiprows=1, delimiter = ";", dtype=str)


current_directory = os.getcwd()
path = 'outputdata'
directory = current_directory + '/' + path + '/'

# parameters
# fiber data
# fiber bead radius 
a = float(f[0])
# number of beads of the fiber                                                    
Nb = int(f[1])
# length                                 
LS = Nb * 2 * a
# aspect ratio                                                     
kappa_inv = LS / (2 * a)
# viscosity
mu = float(f[2])
# weight per unit length                                                     
W = float(f[3])
# elasto-gravitational number                                                        
Be = float(f[4])
# Young's modulus \times the second moment of inertia
EI_product = (W * LS**3) / Be
# bending modulus
B = EI_product
# stretching modulus
S = (4 * EI_product) / a**2
# stretching coefficient
coef_stretching = S / (2 * a)
# bending coefficient
coef_bending = B / (2 * a)

# radius array
radi_vec = np.ones(Nb)
radi_vec = a * radi_vec

# computation time
unit_time_sedimentation = LS * mu / W
number_of_steps_per_unit_time = int(f[5])
number_of_set_time = int(f[6])
dt = unit_time_sedimentation/number_of_steps_per_unit_time
tmin = int(f[7])
tmax = number_of_set_time*unit_time_sedimentation
t= np.arange(tmin, tmax, dt)
TOTAL_STEPS = len(t)

# set the initial vector position
X_0 = np.zeros((Nb, 3), dtype=float)
# define horizontally
X_0[:,0] = np.arange(0, LS, 2 * a)

# set the vector position
X = np.zeros((Nb, 3), dtype=float)

# set the velocity vector
V = np.zeros((Nb, 3), dtype=float)

# set the force vector
F = np.zeros((Nb, 3), dtype=float)

# initialization
X = X_0

# velocity and force arrays at each time step
V_fiber = np.zeros((TOTAL_STEPS, 3 * Nb), dtype=float)

# set the initial values as input of the solver
initial_values = np.squeeze(np.asarray(X.reshape(1, 3 * Nb)))

# set the arguments 
args = (F, a, mu, Nb, LS, coef_bending, coef_stretching)

def bead_velocities(t, X, args):
    # print the simulation time
    print(t)
    
    global F
    F, a, mu, Nb, LS, coef_bending, coef_stretching = args
    
    # get the number of beads
    N = F.shape[0]
    # compute the external forces
    # bending forces
    FB = fo.bending_forces_numba(X.reshape(N, 3), coef_bending, 0)
    # stretching forces
    FS = fo.stretching_forces_numba(X.reshape(N, 3), a, coef_stretching, 0)
    # gravity forces
    F[:,1] = -(W * LS / N ) * np.ones((1, N), dtype=float)
    # compute sum of external forces applied on all beads
    F = F + FB + FS
    # compute the mobility matrix
    M_beads = mob.mobility_matrix_same_bead_size(X.reshape(N, 3), mu, a)
    # compute the bead velocities
    V[:,:] = (np.dot(M_beads, F.reshape(3 * N , 1))).reshape(N, 3)
    
    return np.squeeze(np.asarray(V.reshape(1, 3 * N)))

# BDF solver   
y, V_fiber, all_t = sv.BDF_method(bead_velocities, initial_values, t, dt, \
                                  V_fiber, Nb, V, args)
    
# create new directory
if not os.path.exists(path):
    os.mkdir(path)
else :
    shutil.rmtree(path)
    os.mkdir(path)

# save output data
np.savetxt(directory + "outputpositions.csv", y, delimiter=";")
np.savetxt(directory + "outputvelocities.csv", V_fiber, delimiter=";")
np.savetxt(directory + "outputtime.csv", all_t, delimiter=";" )

duration =  datetime.now() - startTime   
print("time :\t" + "execution\t" + str(duration))



    
liste_BE = []
i=1
while i<=3 :
    liste_2 = [(x+1)*10**i for x in range(9)]
    liste_BE = liste_BE + liste_2
    i=i+1







final_deflection = 100
def_rel =100
for Be in liste_BE :
    while abs(def_rel)>10**-3:
        y_min = min(y)
        y_max = max(y)
        deflection_new = y_max - y_min
        def_rel = final_deflection - deflection_new  
        final_deflection = deflection_new
    
