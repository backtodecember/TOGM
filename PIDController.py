from klampt.io import loader
from klampt import vis,WorldModel
from robosimian_GM_simulation import robosimianSimulator
from copy import copy,deepcopy
from robosimian_wrapper import robosimian
import time
import numpy as np
import math
from copy import copy
#semi-euler integration:
kp = np.array([1000.0,1000.0,1000.0]*4)
ki = np.array([1]*12)
kd = np.array([10.0,8.0,10.0]*4)

#euler integration:
# kp = np.array([400.0,400.0,300.0]*4)
# ki = np.array([0.0,0.0,1.0]*4)
# kd = np.array([4.0,4.0,3.0]*4)


dt = 0.005
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

q_2D[0:3] = [0,0.75,-0.17]
q_desried_0 = deepcopy(q_2D[3:15])
q_2D = np.array(q_2D)[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T
terrain = 1
simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = dt, dyn = 'diffne',print_level = 0,augmented = True,extrapolation = True, \
	integrate_dt = dt,terrain = terrain)

def target_q(time):
	settle_time = 1.0
	if time <= settle_time:
		return q_desried_0
	else:
		return trajectory.eval(time-settle_time,True)[3:15]

	#q = q_2D_0[3:15]
	return q 

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
time.sleep(0.1)

time.sleep(1)

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
	# print('time',simulation_time)
	# print('desired q',desired_q)
	# print('current q',current_q)
	# print('error',error)
	# print('u:',u)
	#print('current_time',simulation_time)

	simulate_start_time = time.time()
	simulator.simulateOnce(u,continuous_simulation = True)
	#print('Simulate Once took:',time.time() - simulate_start_time)
	robot.set_q_2D_(simulator.getConfig())
	vis.unlock()
	time.sleep(0.005)

while vis.shown():
	time.sleep(1)
vis.kill()

# No = 15
# np.save('results/PID_trajectory/'+str(No)+'/q_history.npy',np.array(q_history))
# np.save('results/PID_trajectory/'+str(No)+'/q_dot_history.npy',np.array(q_dot_history))
# np.save('results/PID_trajectory/'+str(No)+'/u_history.npy',np.array(u_history))
# np.save('results/PID_trajectory/'+str(No)+'/time_history.npy',np.array(time_history))

#simulator.closePool()
#vis.kill()