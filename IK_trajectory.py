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
simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = 0.005, dyn = 'own',print_level = 1,augmented = True)

# [0.29677710932029083,-0.9740299999747613]
# [-0.2897485140753952,-0.9740299999954183]
# [-0.2932671869885052,-0.9740299999747615]
# [0.2932584366885086,-0.9740299999954185]


def do_IK(target_position,target_R):
	target = [0,0,0]
	for i in range(4):
		t = to_3D(target_position[i],i)
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


#### This is for trotting 
# q1 = do_IK([back[0],front[1],back[2],front[3]],R)
# q2 = do_IK([mid_high1[0],mid_low1[1],mid_high2[2],mid_low1[3]],R)
# q3 = do_IK([mid_high2[0],mid_low2[1],mid_high1[2],mid_low2[3]],R)
# q4 = do_IK([front[0],back[1],front[2],back[3]],R)
# q5 = do_IK([mid_low1[0],mid_high2[1],mid_low1[2],mid_high1[3]],R)
# q6 = do_IK([mid_low2[0],mid_high1[1],mid_low2[2],mid_high2[3]],R)
# q7 = q1

#### This is for the robot to move its limbs in place, on flat terrain
# low = [[0.52+offset, -0.77403],[-0.45-offset,-0.77403],[-0.45-offset,-0.77403],[0.52+offset, -0.77403]]
# high = [[0.52+offset, -0.52403],[-0.45-offset,-0.52403],[-0.45-offset,-0.52403],[0.52+offset, -0.52403]]

#this is for the slope
height_offset = 0.1
# low = [[0.52+offset, -0.77403 + height_offset],[-0.45-offset*5,-0.77403+ height_offset],[-0.45-offset*5,-0.77403+ height_offset],[0.52+offset, -0.77403+ height_offset]]
# high = [[0.52+offset, -0.52403+ height_offset],[-0.45-offset*5,-0.52403+ height_offset],[-0.45-offset*5,-0.52403+ height_offset],[0.52+offset, -0.52403+ height_offset]]
low = [[0.52+offset*3, -0.77403 + height_offset],[-0.45-offset,-0.77403+ height_offset],[-0.45-offset,-0.77403+ height_offset],[0.52+offset*3, -0.77403+ height_offset]]
high = [[0.52+offset*3, -0.52403+ height_offset],[-0.45-offset,-0.52403+ height_offset],[-0.45-offset,-0.52403+ height_offset],[0.52+offset*3, -0.52403+ height_offset]]

q1 = do_IK([low[0],low[1],low[2],low[3]],R)
q2 = do_IK([high[0],low[1],high[2],low[3]],R)
q3 = do_IK([high[0],low[1],high[2],low[3]],R)
q4 = do_IK([low[0],low[1],low[2],low[3]],R)
q5 = do_IK([low[0],high[1],low[2],high[3]],R)
q6 = do_IK([low[0],high[1],low[2],high[3]],R)
q7 = q1


#interpolate the trajectory
total_time = 9
N_of_steps = 3
total_milestones = 600
segment_milestones = 100
dt = 0.015/N_of_steps
u = np.linspace(0,1,segment_milestones+1).tolist()
 


traj = []
times = []
current_time = 0.0

##repeat the full step N times
for iter in range(N_of_steps):
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

#save the trajectory
trajectory = Trajectory(times = times, milestones = traj)
loader.save(trajectory,'auto','data/gallop_gait_3_steps_slope2')
exit()

#visualize the robot
world = simulator.getWorld()
vis.add("world",world)
vis.show()
start_time = time.time()
vis_dt = 0.05
q = trajectory.eval(0.0)
simulator.reset(np.array(q))
time.sleep(2)
while(time.time() - start_time < total_time):
	vis.lock()
	vis.addText('time','time: '+str(current_time))
	current_time = time.time() - start_time
	q = trajectory.eval(current_time)
	simulator.reset(np.array(q))
	vis.unlock()
	time.sleep(vis_dt)

while vis.shown():
	time.sleep(dt)
vis.kill()
