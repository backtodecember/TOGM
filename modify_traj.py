"""
This file is take a optimized trajectory and rotate it for a different slope
"""

import numpy as np
from robosimian_GM_simulation import robosimianSimulator
import matplotlib.pyplot as plt
import configs
from copy import deepcopy
from robosimian_wrapper import robosimian
from klampt import vis
import time
from klampt.math import so2
import math
Xs = np.load('results/41/run1_test41/solution_x918.npy')
Us = np.load('results/41/run1_test41/solution_u918.npy')
ts = np.load('results/PID_trajectory/32/time_init_guess.npy')
(N,D) = np.shape(Xs)
angle = 10.0/180.0*math.pi
for i in range(N):
    position = Xs[i,0:2].tolist()
    velocity = Xs[i,15:17].tolist()
    rotated_pos = so2.apply(-angle,position)
    rotated_vel = so2.apply(-angle,velocity)
    Xs[i,0] = rotated_pos[0]
    Xs[i,1] = rotated_pos[1]
    Xs[i,15] = rotated_vel[0]
    Xs[i,16] = rotated_vel[1]

path = "results/PID_trajectory/33/"
np.save(path+'q_init_guess.npy',Xs[:,0:15])
np.save(path+'q_dot_init_guess.npy',Xs[:,15:30])
np.save(path+'u_init_guess.npy',Us)
np.save(path+'time_init_guess.npy',ts)




