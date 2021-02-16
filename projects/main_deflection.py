#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:40:50 2021

@author: gabriel
"""
import sys
import numpy as np
import os
from datetime import datetime


sys.path.append('.')
sys.path.append('../')


final_deflection = 100

startTime = datetime.now()

# load  input file
f = np.loadtxt("inputdata.csv", skiprows=1, delimiter = ";", dtype=str)     

current_directory = os.getcwd()
path = 'outputdata'
directory = current_directory + '/' + path + '/'

y = np.loadtxt(directory + "outputpositions.csv", delimiter=";")
tk = np.loadtxt(directory + "outputtime.csv", delimiter=";")
TOTAL_STEPS = y.shape[0]

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
unit_time_sedimentation = LS * mu / W

# define the domain
Lim = 0.5







#final_deflection = 1000
deflection_ante=0

liste_BE = []
i=1
while i <=3 :
    liste_2 = [(x+1)*10**i for x in range(10) ]
    liste_BE = liste_BE + liste_2
    i=i+1
    
for k in range(TOTAL_STEPS - 1):
    
    all_x = list()
    all_y = list()
    
    all_f_x = list()
    all_f_y = list()
    
    Ny = int(y.shape[1]/3)
    for kk in range(Ny) :
            all_x.append(y[:,(3 * kk)][k])
            all_y.append(y[:,3 * kk+1][k])
            
            all_x[:] = [x / LS for x in all_x]
            all_y[:] = [y / LS for y in all_y]
#            print(all_y)
            y_min = min(all_y)
            y_max = max(all_y)
            
            deflection = y_max - y_min
            print(deflection)
    def_rel = abs(deflection - deflection_ante)
    #print(def_rel)
    if (def_rel)<10**-2 and def_rel != 0.0:
        #print(deflection)
        #print(def_rel)
        print("ok")
        final_deflection=deflection
    deflection_ante=deflection
print(final_deflection)               
                
         

