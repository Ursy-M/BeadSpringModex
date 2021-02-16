#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 01:20:28 2021

@author: gabriel
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

# load ouput data
y = np.loadtxt(directory + "outputpositions.csv", delimiter=";")
tk = np.loadtxt(directory + "outputtime.csv", delimiter=";")

# get the number of time step
TOTAL_STEPS = y.shape[0]

deflection = 0
deflection_relative = 1
k = 1
eps = 10^-2

all_deflection = list()
while k < (TOTAL_STEPS - 1) and deflection_relative > eps :
    for k in range(1,TOTAL_STEPS - 1):
        deflection_anterieur = deflection
        print("ok")
    #    if deflection_relative == 0.0 :
    #        k += 1
    #    else :
        all_x = list()
        all_y = list()
        
        
        Ny = int(y.shape[1]/3)
        for kk in range(Ny) :
            
            all_x.append(y[:,(3 * kk)][k])
            all_y.append(y[:,3 * kk+1][k])
            
        all_x[:] = [x / LS for x in all_x]
        all_y[:] = [y / LS for y in all_y]
    #           print(all_y)
        y_min = min(all_y)
        y_max = max(all_y)
            
        deflection = y_max - y_min
        print(deflection)
        deflection_relative = abs((deflection - deflection_anterieur)/deflection) # déflection relative
        #print(deflection_relative)
        print(deflection)
        k = k + 1
        
                
all_deflection.append(deflection)            
            
            
            
            
            
            