"""
This file takes the result for a trajectory and animate it
The ground reaction force is calculated with the state and control at that node
"""

import numpy as np
from robosimian_GM_simulation import robosimianSimulator
import matplotlib.pyplot as plt
import configs
from copy import deepcopy
from klampt import vis
from klampt.model import trajectory
from klampt.math import vectorops as vo
import time

dt = 0.005
q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
q_dot0 = np.zeros((15,1))
robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = dt, solver = 'cvxpy', augmented = True)

case = '11-2'
u = np.load('results/'+case+'/solution_u.npy')
x_simulation = np.load('results/'+case+'/solution_x.npy')
(N,_) = np.shape(x_simulation)
u_traj = u

#slow down 20x
vis_dt = 0.005*20
force_scale = 0.001 #200N would be 0.2m

world = robot.getWorld()
vis.add("world",world)
vis.show()
time.sleep(1)
#while vis.shown():
for i in range(N):
	vis.lock()
	robot.reset(x_simulation[i,0:15],x_simulation[i,15:30])
	ankle_positions = robot.robot.get_ankle_positions(full = True)
	force,accel = robot.simulateOnce(u[i],continuous_simulation = False, SA = False, fixed = False)
	for i in range(4):
		force_vector = vo.mul(force[0+i*3:2+i*3],force_scale)
		limb_force = trajectory.Trajectory(times = [0,1],milestones = [vo.add(ankle_positions[i][0:3],[0,-0.1,0]),\
			vo.add(vo.add(ankle_positions[i][0:3],[force_vector[0],0,force_vector[1]]),[0,-0.1,0])])
		vis.add('force'+str(i+1),limb_force)
	vis.unlock()
	time.sleep(vis_dt)
vis.kill()