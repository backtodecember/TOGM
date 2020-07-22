"""
This file is to visualize the ankle positions and ground reaction force for a trajectory

"""

import numpy as np
from robosimian_GM_simulation import robosimianSimulator
import matplotlib.pyplot as plt
import configs
from copy import deepcopy
from robosimian_wrapper import robosimian
from klampt import vis
import time
# dt = 0.005
# q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
# q_dot0 = np.zeros((15,1))
# robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = dt, solver = 'cvxpy', augmented = True)

# N = 50
# case = '11-2'
# u = np.load('results/'+case+'/solution_u.npy') #both u and x are N-by-12/30 numpy matrices
# x = np.load('results/'+case+'/solution_x.npy')

# limb = 4

######## plot the ankle positions #########
# iterations = []
# ps = []
# counter = 0
# #plot the end effector positions...
# for q,q_dot in zip(x[:,0:15],x[:,15:30]):
# 	robot.reset(q,q_dot)
# 	p = robot.robot.get_ankle_positions()
# 	ps.append(p[limb-1])
# 	iterations.append(counter)
# 	counter += 1
# ps = np.array(ps)
# iterations = np.array(iterations)
# fig,axs = plt.subplots(3,1,sharex=True)
# axs[0].plot(iterations,ps[:,0])
# axs[0].set_title('x-position')
# axs[0].set(ylabel='m')
# axs[1].plot(iterations,ps[:,1])
# axs[1].set_title('z-position')
# axs[1].set(ylabel='m')
# axs[2].plot(iterations,ps[:,2])
# axs[2].set_title('angle') 
# axs[2].set(xlabel='iterations', ylabel='rad')
# fig.suptitle('Limb ' + str(limb) +' Ankle Poses')
# plt.show()


######## plot the ground reaction forces by simulating#########
# iterations = []
# counter = 0
# ps1 = []
# ps2 = []
# ps3 = []
# ps4 = []
# #plot the ground reaction forces
# for q,q_dot,one_u in zip(x[:,0:15],x[:,15:30],u):
# 	if counter == 0:
# 		robot.reset(q,q_dot)
# 	ground_force,_ = robot.simulateOnce(one_u,continuous_simulation = True)
# 	if counter == 0:
# 		data = np.array(ground_force)
# 	else:
# 		data = np.hstack((data,ground_force))
# 	iterations.append(counter)
# 	counter += 1
# 	#print('netforce:',ground_force[0:3]+ground_force[3:6]+ground_force[6:9]+ground_force[9:12])
# 	ps1.append(robot.robot.get_ankle_positions()[0])
# 	ps2.append(robot.robot.get_ankle_positions()[1])
# 	ps3.append(robot.robot.get_ankle_positions()[2])
# 	ps4.append(robot.robot.get_ankle_positions()[3])

# iterations = np.array(iterations)
# ps1 = np.array(ps1)
# ps2 = np.array(ps2)
# ps3 = np.array(ps3)
# ps4 = np.array(ps4)

# #print(np.shape(iterations),np.shape(data),np.shape(data[0,:]))
# for limb in range(4):
# 	fig,axs = plt.subplots(3,1,sharex=True)
# 	axs[0].plot(iterations,data[limb*3+0,:])
# 	axs[0].set_title('x-force')
# 	axs[0].grid()
# 	axs[0].set(ylabel='N')
# 	axs[1].plot(iterations,data[limb*3+1,:])
# 	axs[1].set_title('z-force')
# 	axs[1].set(ylabel='N')
# 	axs[1].grid()
# 	axs[2].plot(iterations,data[limb*3+2,:])
# 	axs[2].set_title('torque') 
# 	axs[2].set(xlabel='iterations', ylabel='Nm')
# 	axs[2].grid()
# 	fig.suptitle('Limb ' + str(limb+1) +' Ground Reaction Force ')
# 	plt.show()

# fig,axs = plt.subplots(3,1,sharex=True)
# axs[0].plot(iterations,ps1[:,0])
# axs[0].set_title('x-position')
# axs[0].set(ylabel='m')
# axs[1].plot(iterations,ps1[:,1])
# axs[1].set_title('z-position')
# axs[1].set(ylabel='m')
# axs[2].plot(iterations,ps1[:,2])
# axs[2].set_title('angle') 
# axs[2].set(xlabel='iterations', ylabel='rad')
# fig.suptitle('Limb 1 Ankle Poses')
# plt.show()

# fig,axs = plt.subplots(3,1,sharex=True)
# axs[0].plot(iterations,ps2[:,0])
# axs[0].set_title('x-position')
# axs[0].set(ylabel='m')
# axs[1].plot(iterations,ps2[:,1])
# axs[1].set_title('z-position')
# axs[1].set(ylabel='m')
# axs[2].plot(iterations,ps2[:,2])
# axs[2].set_title('angle') 
# axs[2].set(xlabel='iterations', ylabel='rad')
# fig.suptitle('Limb 2 Ankle Poses')
# plt.show()

# fig,axs = plt.subplots(3,1,sharex=True)
# axs[0].plot(iterations,ps3[:,0])
# axs[0].set_title('x-position')
# axs[0].set(ylabel='m')
# axs[1].plot(iterations,ps3[:,1])
# axs[1].set_title('z-position')
# axs[1].set(ylabel='m')
# axs[2].plot(iterations,ps3[:,2])
# axs[2].set_title('angle') 
# axs[2].set(xlabel='iterations', ylabel='rad')
# fig.suptitle('Limb 3 Ankle Poses')
# plt.show()

# fig,axs = plt.subplots(3,1,sharex=True)
# axs[0].plot(iterations,ps4[:,0])
# axs[0].set_title('x-position')
# axs[0].set(ylabel='m')
# axs[1].plot(iterations,ps4[:,1])
# axs[1].set_title('z-position')
# axs[1].set(ylabel='m')
# axs[2].plot(iterations,ps4[:,2])
# axs[2].set_title('angle') 
# axs[2].set(xlabel='iterations', ylabel='rad')
# fig.suptitle('Limb 4 Ankle Poses')
# plt.show()


###### quickly visualize x 
x = np.load('results/27/run3/solution_x961.npy')
# x = np.load('temp_files/solution_x891.npy')
robot = robosimian()
world = robot.get_world()
robot.set_q_2D_(x[0,0:15])
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
	robot.set_q_2D_(x[i,0:15])
	vis.addText('time','time: '+str(current_time))
	current_time += dt
	vis.unlock()
	time.sleep(0.1)

while vis.shown():
	time.sleep(1)

vis.kill()