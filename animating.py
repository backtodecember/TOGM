import numpy as np
from robosimian_GM_simulation import robosimianSimulator
import matplotlib.pyplot as plt
import configs
from copy import deepcopy
from klampt import vis
import time
q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
q_dot0 = np.zeros((15,1))
robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = 0.005, solver = 'cvxpy', augmented = True)

N = 50
case = '11-2'
u = np.load('results/'+case+'/solution_u.npy')
x_simulation = np.load('results/'+case+'/solution_x.npy')
print(np.shape(x_simulation))
u_traj = u

dt = 0.005*20


world = robot.getWorld()
robot_model = robot.getRobot()

vis.add("world",world)

vis.show()
time.sleep(10)
#while vis.shown():
for i in range(N):
	vis.lock()
	robot_model.set_q_2D_(x_simulation[i,:])
	vis.unlock()
	time.sleep(dt)
vis.kill()