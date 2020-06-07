from klampt.io import loader
from klampt import vis
from robosimian_GM_simulation import robosimianSimulator
from copy import copy,deepcopy
import time
import numpy as np

kp = np.array([1000.0,1000.0,1000.0]*4)
ki = np.array([2.0]*12)
kd = np.array([10.0,8.0,10.0]*4)
dt = 0.005
trajectory = loader.loadTrajectory('data/trotting_gait')
total_time = trajectory.endTime()

q_2D = trajectory.eval(0.0)
q_2D[0:3] = [0,0.90,0]
q_desried_0 = deepcopy(q_2D[3:15])
q_2D = np.array(q_2D)[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T
simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = dt, solver = 'cvxpy',print_level = 0,augmented = True,extrapolation = True)

def target_q(time):
	settle_time = 1.0
	if time <= settle_time:
		return q_desried_0
	else:
		return trajectory.eval(time-settle_time,True)[3:15]

#print(target_q(0.2),target_q(1.6))

#start simulation
error = np.array([0.0]*12)
last_error = np.array([0.0]*12)
accumulated_error = np.array([0.0]*12)
world = simulator.getWorld()
vis.add("world",world)
vis.show()
vis.addText('time','time: '+str(0))
time.sleep(1.0)
simulation_time = 0.0
start_time = time.time()


q_history = []
q_dot_history = []
u_history = []
time_history = []


while vis.shown() and (simulation_time < 10.001):
	#loop_start_time = time.time()
	vis.lock()
	#simulation_time = time.time() - start_time
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
	# if simulation_time > 0.02:
	# 	u[11] = 5000.0
	simulation_time += dt
	vis.clearText()
	vis.addText('time','time: '+str(simulation_time))
	print('time',simulation_time)
	print('desired q',desired_q)
	print('current q',current_q)
	print('error',error)
	print('u:',u)
	#print('current_time',simulation_time)

	simulate_start_time = time.time()
	simulator.simulateOnce(u,continuous_simulation = True)
	print('Simulate Once took:',time.time() - simulate_start_time)
	vis.unlock()
	time.sleep(0.001)


vis.kill()

np.save('results/PID_trajectory/2/q_history.npy',np.array(q_history))
np.save('results/PID_trajectory/2/q_dot_history.npy',np.array(q_dot_history))
np.save('results/PID_trajectory/2/u_history.npy',np.array(u_history))
np.save('results/PID_trajectory/2/time_history.npy',np.array(time_history))