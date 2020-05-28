"""
This file design an IK trajectory

"""
import math
from robosimian_GM_simulation import robosimianSimulator
from klampt.math import vectorops as vo
from klampt import vis
import numpy as np
import time
from klampt.model.trajectory import Trajectory
from klampt.io import loader
y_positions = [-0.5603823784573472,-0.5603824347078008,0.5603999999913027,0.5604000000062257]
R = [0,0,-1,0,1,0,1,0,0] #transform of the ankles when they are straight down
def to_3D(p,limb):
	return [p[0],y_positions[limb],p[1]]

q_2D = np.array([0,0,0,-math.pi/2.0,-0.1,0.1,math.pi/2,0.1,-0.1,-math.pi/2.0,-0.1,0.1,math.pi/2,0.1,-0.1])[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T
simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = 0.005, solver = 'cvxpy',print_level = 1,augmented = True)

# [0.29677710932029083,-0.9740299999747613]
# [-0.2897485140753952,-0.9740299999954183]
# [-0.2932671869885052,-0.9740299999747615]
# [0.2932584366885086,-0.9740299999954185]


def do_IK(target_position,target_R):
	target = [0,0,0]
	for i in range(4):
		t = to_3D(target_position[i],i)
		#print(t)
		q = simulator.robot.do_IK(target_R[i],t,i)
		if q == None:
			print(i)
		target = target + q[3+i*3+0:3+i*3+3]
	return target

R1 = [7.346042795959517e-06, 7.347223069371793e-08, -0.9999999999730153, -0.0100024332041339, \
	0.9999499744137188, -9.747178703479909e-12, 0.9999499743867355, 0.010002433203864057, 7.346410206874145e-06]
R2 = [9.708807534924425e-14, -9.746695459416838e-12, -1.0000000000000002, -0.0100024332041339, 0.9999499744137188, -9.747178703479909e-12, 0.999949974413719, 0.0100024332041339, -4.74979982603834e-16]
R3 = [-7.346410207274962e-06, -9.747179572067733e-12, -0.9999999999730155, 2.6535897933348288e-06, -0.9999999999964793, -9.747179571905247e-12, -0.9999999999694947, -2.6535897933348288e-06, 7.346410207274962e-06]
R4 = [4.8299840918473434e-17, 9.747179572067733e-12, -1.0000000000000004, 2.6535897933348288e-06, -0.9999999999964793, -9.747179571905247e-12, -0.9999999999964796, -2.6535897933348288e-06, -7.416483899604634e-17]
R = [R1,R2,R3,R4]


#use these cartesian EE positions
offset = 0.05
front = [[0.6968, -0.77403],[-0.1897-0.1,-0.77403],[-0.1933-0.1,-0.77403],[0.6968, -0.77403]]
# mid_high = [[0.4368+offset, -0.52403],[-0.4397-offset,-0.52403],[-0.4433-offset,-0.52403],[0.4368+offset, -0.52403]]
# mid_low = [[0.4368+offset, -0.77403],[-0.4397-offset,-0.77403],[-0.4433-offset,-0.77403],[0.4368+offset, -0.77403]]
mid_high1 = [[0.2+offset, -0.52403],[-0.2-offset,-0.52403],[-0.2-offset,-0.52403],[0.2+offset, -0.52403]]
mid_high2 = [[0.7+offset, -0.52403],[-0.7-offset,-0.52403],[-0.7-offset,-0.52403],[0.7+offset, -0.52403]]
# mid_low = [[0.4368+offset, -0.77403],[-0.4397-offset,-0.77403],[-0.4433-offset,-0.77403],[0.4368+offset, -0.77403]]
mid_low1 = [[0.52+offset, -0.77403],[-0.45-offset,-0.77403],[-0.45-offset,-0.77403],[0.52+offset, -0.77403]]
mid_low2 = [[0.38+offset, -0.77403],[-0.6-offset,-0.77403],[-0.6-offset,-0.77403],[0.38+offset, -0.77403]]
back = [[0.1968+0.1, -0.77403],[-0.6897,-0.77403],[-0.6933,-0.77403],[0.1968+0.1, -0.77403]]


#### This is for initially playing around with things
# q1 = do_IK([back[0],front[1],back[2],front[3]],R)
# q2 = do_IK([mid_high[0],mid_low[1],mid_high[2],mid_low[3]],R)
# q3 = do_IK([front[0],back[1],front[2],back[3]],R)
# q4 = do_IK([mid_low[0],mid_high[1],mid_low[2],mid_high[3]],R)
# q5 = q1

q1 = do_IK([back[0],front[1],back[2],front[3]],R)
q2 = do_IK([mid_high1[0],mid_low1[1],mid_high2[2],mid_low1[3]],R)
q3 = do_IK([mid_high2[0],mid_low2[1],mid_high1[2],mid_low2[3]],R)
q4 = do_IK([front[0],back[1],front[2],back[3]],R)
q5 = do_IK([mid_low1[0],mid_high2[1],mid_low1[2],mid_high1[3]],R)
q6 = do_IK([mid_low2[0],mid_high1[1],mid_low2[2],mid_high2[3]],R)
q7 = q1

#simulator.debugSimulation()
#print('ankle positions:',simulator.robot.get_ankle_positions())

# print(q1,q2,q3)

#interpolate the trajectory
total_time = 9
total_milestones = 600
segment_milestones = 100
dt = 0.015
u = np.linspace(0,1,segment_milestones+1).tolist()
 
traj = []
times = []
current_time = 0.0
for i in range(segment_milestones):
	traj.append(vo.interpolate(q1,q2,u[i])) #interpolate includes both ends
	times.append(current_time)
	current_time += dt

for i in range(segment_milestones):
	traj.append(vo.interpolate(q2,q3,u[i])) #interpolate includes both ends
	times.append(current_time)
	current_time += dt
for i in range(segment_milestones):
	traj.append(vo.interpolate(q3,q4,u[i])) #interpolate includes both ends
	times.append(current_time)
	current_time += dt
for i in range(segment_milestones):
	traj.append(vo.interpolate(q4,q5,u[i])) #interpolate includes both ends
	times.append(current_time)
	current_time += dt
for i in range(segment_milestones):
	traj.append(vo.interpolate(q5,q6,u[i])) #interpolate includes both ends
	times.append(current_time)
	current_time += dt
for i in range(segment_milestones):
	traj.append(vo.interpolate(q6,q7,u[i])) #interpolate includes both ends
	times.append(current_time)
	current_time += dt

traj.append(vo.interpolate(q6,q7,u[segment_milestones])) #interpolate includes both ends
times.append(current_time)

trajectory = Trajectory(times = times, milestones = traj)

loader.save(trajectory,'auto','data/trotting_gait')
world = simulator.getWorld()
vis.add("world",world)
vis.show()
start_time = time.time()
while(time.time() - start_time < total_time):
	vis.lock()
	current_time = time.time() - start_time
	q = trajectory.eval(current_time)
	simulator.reset(np.array(q))
	vis.unlock()
	time.sleep(dt)

while vis.shown():
	time.sleep(dt)
vis.kill()



#This is for designing trajectories with splines, not really used...
# import numpy as np
# import scipy.interpolate as interpolate
# import matplotlib.pyplot as plt
# import scipy.interpolate as si


# def scipy_bspline(cv, degree=3, periodic=False):
#     """ Calculate n samples on a bspline
#         cv :      Array ov control vertices
#         n  :      Number of samples to return
#         degree:   Curve degree
#         periodic: True - Curve is closed # else curve is open
#         https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy - source
#     """
#     cv = np.asarray(cv)  # makes an array of cv if not already an array
#     count = cv.shape[0]  # returns number of points in array

#     # Closed curve
#     if periodic:
#         kv = np.arange(-degree, count + degree + 1)  # sets up numbers necessary for b spline
#         factor, fraction = divmod(count + degree + 1, count)
#         cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)), -1, axis=0)
#         degree = np.clip(degree, 1, degree)

#     # Opened curve # we want opened curve
#     else:
#         degree = np.clip(degree, 1, count - 1)
#         kv = np.clip(np.arange(count + degree + 1) - degree, 0,
#                      count - degree)  # sets up numbers necessary for b spline clamped knot

#     # Return samples
#     max_param = count - (degree * (1 - periodic))
#     spl = si.BSpline(kv, cv, degree)
#     # return spl(np.linspace(0, max_param, n))
#     return spl, max_param

# x = np.array([0.1968,0.4468,0.6968])
# y = np.array([-0.77403,-0.52403,-0.77403])


# trajectory = [[0.1968,-0.77403],[0.4368,-0.32403],[0.6968,-0.77403]]#,[0.1968,-0.77403]]

# spline, bsplParam1 = scipy_bspline(trajectory, degree = 10, periodic = True)
# us = np.linspace(0,1,30).tolist()
# ys = []

# for u in us:
# 	ys.append(spline(bsplParam1*u).tolist())
# ys = np.array(ys)
# plt.plot(x, y, 'bo', label='Original points')
# plt.plot(ys[:,0], ys[:,1], 'r', label='BSpline')
# plt.grid()
# plt.legend(loc='best')
# plt.axis('equal')
# plt.show()


#cartesian trajectory for  