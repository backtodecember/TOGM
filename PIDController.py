from klampt.io import loader
from klampt import vis,WorldModel
from robosimian_GM_simulation import robosimianSimulator
from copy import copy,deepcopy
from robosimian_wrapper import robosimian
import time
import numpy as np
import math
from copy import copy
from klampt.model import trajectory as trajLib
from scipy.spatial import ConvexHull
from klampt.math import vectorops as vo
#semi-euler integration:
kp = np.array([2000.0,2000.0,2000.0]*4)
ki = np.array([20]*12)
kd = np.array([12.0,12.0,12.0]*4)

#euler integration:
# kp = np.array([400.0,400.0,300.0]*4)
# ki = np.array([0.0,0.0,1.0]*4)
# kd = np.array([4.0,4.0,3.0]*4)


dt = 0.005
# trajectory = loader.loadTrajectory('data/gallop_gait_3_steps_slope')
# trajectory = loader.loadTrajectory('data/gallop_gait_3_steps_slope2')
trajectory = loader.loadTrajectory('data/gallop_gait_3_steps')
total_time = trajectory.endTime()

q_2D = trajectory.eval(0.0)
# q_2D = [0,0,0] + [-math.pi*0.9/2,math.pi/2,-math.pi*1.1/2,math.pi*0.9/2,-math.pi/2,math.pi*1.1/2,-math.pi*0.9/2,math.pi/2,-math.pi*1.1/2,math.pi*0.9/2,\
# 	-math.pi/2,math.pi*1.1/2]

# diff = 0.75
# q_2D_0 = [0,0,0] + [-7.25839573e-01 + diff, -1.76077022e+00 - diff ,8.56369518e-01,\
# 	7.30855272e-01 - 0.15,1.76897237 - 0.1,-8.53873601e-01,\
# 	-3.90820203e-01 + diff,-6.04853525e-01 - diff,-4.99734554e-01,\
# 	4.00930729e-01,5.78514982e-01,5.18754554e-01]

# q_2D = copy(q_2D_0)

degree10 = -math.pi*10.0/180.0
# degree10 = 0.0
q_2D[0:3] = [0,0.95,-degree10*0.75]
q_2D[5] -= degree10 
q_2D[8] -= degree10 
q_2D[11] += degree10
q_2D[14] += degree10

q_desired_0 = deepcopy(q_2D[3:15])

q_2D = np.array(q_2D)[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T

terrain = 0

simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = dt, dyn = 'own',print_level = 0,augmented = True,extrapolation = True, \
	integrate_dt = dt,terrain = terrain,diffne_mu = 1e-6)

def target_q(time):
	settle_time = 1.0
	if time <= settle_time:
		return q_desired_0
	else:
		t = time-settle_time
		# if t > 4:
		# 	t = 4

		tmp = trajectory.eval(t,True)[3:15]
		tmp[2] -= degree10
		tmp[5] -= degree10
		tmp[8] += degree10
		tmp[11] += degree10
		return tmp
		
	#q = q_2D_0[3:15]
	#return q 

#print(target_q(0.2),target_q(1.6))

#start simulation
error = np.array([0.0]*12)
last_error = np.array([0.0]*12)
accumulated_error = np.array([0.0]*12)

robot = robosimian(terrain = terrain)
world = robot.get_world()
robot.set_q_2D_(q_2D)

vis.add("world",world)
vis.show()
vis.addText('time','time: '+str(0))
time.sleep(15.01)

# vis.kill()
# exit()
# time.sleep(5)

simulation_time = 0.0
start_time = time.time()


q_history = []
q_dot_history = []
u_history = []
time_history = []

WS_vis = True
counter1 = 0
counter2 = 0
while vis.shown() and (simulation_time < 10.001):
	#loop_start_time = time.time()
	vis.lock()
	#simulation_time = time.time() - start_time
	#print(simulator.getConfig())
	current_q = simulator.getConfig()[3:15] #1d array of 15 elements
	desired_q = np.array(target_q(simulation_time))
	last_error = deepcopy(error)
	error = desired_q - current_q
	dError = (error - last_error)/dt
	accumulated_error += error
	u_raw = np.multiply(kp,error) + np.multiply(kd,dError) + np.multiply(ki,accumulated_error)
	u = np.clip(u_raw,-200.0,200.0)

	for i in range(4):
		u[i*3+1] += u[i*3+2] 
		u[i*3] += u[i*3+1]
	u = np.clip(u_raw,-300.0,300.0) 

	#record 
	q_history.append(simulator.getConfig().tolist())
	q_dot_history.append(simulator.getVel().tolist())
	u_history.append(u.tolist())
	time_history.append(simulation_time)

	simulation_time += dt
	vis.clearText()
	vis.addText('time','time: '+str(simulation_time))
	simulate_start_time = time.time()

	if WS_vis:
		#remove the previous WS
		for i in range(counter1):
			vis.remove('line1_'+str(i))
		for i in range(counter2):
			vis.remove('line2_'+str(i))
		##Ws is a 12x104 numpy array, f is [12,1]
		f,_,WS = simulator.simulateOnce(u,continuous_simulation = True,fixed= False,get_WS = True)
		anklePositions = simulator.robot.get_ankle_positions(full = True)
		anklePosition1 = anklePositions[0][0:3]
		anklePosition2 = anklePositions[1][0:3]
		WS1 = WS[0:3,0:26].T
		WS2 = WS[3:6,26:52].T

		print(WS1)
		print(WS2)
		K1 = ConvexHull(WS1).simplices
		K2 = ConvexHull(WS2).simplices
		xscale = 0.001
		zscale = 0.001
		y = -0.5
		x1 = 0.3
		x2 = -0.3
		counter1 = 0
		counter2 = 0
		for tri in K1:
			#only plotting fx and fz here
			pts = [[WS1[tri[0],0]*xscale+x1,y,WS1[tri[0],1]*zscale],\
				[WS1[tri[1],0]*xscale+x1,y,WS1[tri[1],1]*zscale],\
				[WS1[tri[2],0]*xscale+x1,y,WS1[tri[2],1]*zscale],\
				[WS1[tri[0],0]*xscale+x1,y,WS1[tri[0],1]*zscale]]
			for i in range(4):
				pts[i] = vo.add(pts[i],anklePosition1)
			ts = np.linspace(0,1,4)
			cvxTraj = trajLib.Trajectory(ts,pts)
			vis.add('line1_'+str(counter1),cvxTraj)
			vis.setColor('line1_'+str(counter1),0,1,0,1)
			vis.hideLabel('line1_'+str(counter1),True)
			counter1 += 1
		for tri in K2:
			#only plotting fx and fz here
			pts = [[WS2[tri[0],0]*xscale+x2,y,WS2[tri[0],1]*zscale],\
				[WS2[tri[1],0]*xscale+x2,y,WS2[tri[1],1]*zscale],\
				[WS2[tri[2],0]*xscale+x2,y,WS2[tri[2],1]*zscale],\
				[WS2[tri[0],0]*xscale+x2,y,WS2[tri[0],1]*zscale]]
			for i in range(4):
				pts[i] = vo.add(pts[i],anklePosition2)
			ts = np.linspace(0,1,4)
			cvxTraj = trajLib.Trajectory(ts,pts)
			vis.add('line2_'+str(counter2),cvxTraj)
			vis.setColor('line2_'+str(counter2),0,1,0,1)
			vis.hideLabel('line2_'+str(counter2),True)
			counter2 += 1

		force = [vo.add([x1,y,0],anklePosition1),vo.add([f[0]*xscale+x1,y,f[1]*zscale],anklePosition1)]
		cvxTraj = trajLib.Trajectory([0,1],force)
		vis.add('force1',cvxTraj)
		vis.setColor('force1',1,0,0,1)
		vis.hideLabel('force1',True)
		force = [vo.add([x2,y,0],anklePosition2),vo.add([f[3]*xscale+x2,y,f[4]*zscale],anklePosition2)]
		cvxTraj = trajLib.Trajectory([0,1],force)
		vis.add('force2',cvxTraj)
		vis.setColor('force2',1,0,0,1)
		vis.hideLabel('force2',True)
	else:
		simulator.simulateOnce(u,continuous_simulation = True)
	vel = simulator.getVel()

	# if vel[0] > 0.4*math.cos(degree10):
	if vel[0] < 0:
		simulator.q_dot[0,0] = 0
	#print('Simulate Once took:',time.time() - simulate_start_time)
	robot.set_q_2D_(simulator.getConfig())
	vis.unlock()
	time.sleep(0.005)

while vis.shown():
	time.sleep(1)
vis.kill()

No = 31
np.save('results/PID_trajectory/'+str(No)+'/q_history.npy',np.array(q_history))
np.save('results/PID_trajectory/'+str(No)+'/q_dot_history.npy',np.array(q_dot_history))
np.save('results/PID_trajectory/'+str(No)+'/u_history.npy',np.array(u_history))
np.save('results/PID_trajectory/'+str(No)+'/time_history.npy',np.array(time_history))

#simulator.closePool()
#vis.kill()