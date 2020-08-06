"""
This file is to visualize the ankle positions and ground reaction force for a trajectory

"""

import numpy as np
import matplotlib.pyplot as plt
import configs
from copy import deepcopy
from robosimian_wrapper_3D_base import robosimian
from klampt import vis
import time


###### quickly visualize x 
# x = np.load('results/28/run19/solution_x204.npy')
# x= np.load('results/31/run2/solution_x1.npy')
x = np.load('temp_files3/solution_x151.npy')
robot = robosimian()
world = robot.get_world()
robot.set_q_2D_(x[0,0:18])
dt = 0.05

(m,n) = np.shape(x)

current_time = 0.0
vis.add("world",world)
vis.show()
vis.addText('time','time: '+str(current_time))
time.sleep(10.0)
simulation_time = 0.0
start_time = time.time()

for i in range(m):
	vis.lock()
	robot.set_q_2D_(x[i,0:18])
	vis.addText('time','time: '+str(current_time))
	current_time += dt
	vis.unlock()
	time.sleep(0.1)

while vis.shown():
	time.sleep(1)

vis.kill()