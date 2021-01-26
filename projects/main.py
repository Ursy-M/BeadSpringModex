# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:30:06 2021

@author: modex
"""

""" sedimentation of a flexible fiber in a viscous fluid """

import sys
import numpy as np

# try to import numba
try :
    from numba import prange
except ImportError:
    print('numba no found')

sys.path.append('.')
sys.path.append('../')

from mobility_numba import mobility_numba as mob
from forces_numba import forces_numba as fon
from solver import solver as sv


