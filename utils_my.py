import numpy as np
import matplotlib.pyplot as plt
from klampt.math import vectorops as vo
from klampt import vis
import time
from robosimian_wrapper import robosimian
from robosimian_GM_simulation import robosimianSimulator

def trajectory_loader(start,end,dt,number):
	"""
	load a trajectory into desired start and end, with  desired dt

	"""
	if number == 2:
		#1st second is standing still, the rest of 8 second is walking (1,3 swing first, then 2,4 swing)
		original_dt = 0.005
		start_walking  = 1.0
		end_walking = 10.0
		scale = round(dt/original_dt)
		path = "results/PID_trajectory/11/"
		q_history = np.load(path + "q_history.npy")[200:2001]
		q_dot_history = np.load(path + "q_dot_history.npy")[200:2001]
		time_history = np.load(path + "time_history.npy")[200:2001]
		u_history = np.load(path + "u_history.npy")[200:2001]
		multiplier = round(dt/original_dt)
		q = []
		q_dot = []
		time = []
		u = []
		for i in range(len(q_history)):
			if i%multiplier == 0:
				q.append(q_history[i])
				q_dot.append(q_dot_history[i])
				time.append(time_history[i]-start_walking)
				u_total = [0.0]*12
				counter = 0
				for j in range(multiplier):
					if i+j > 1800:
						break 
					u_total = vo.add(u_total,u_history[i+j])
					counter += 1
				u.append(vo.div(u_total,counter))

		# q.append(q_history[-1])
		# q_dot.append(q_dot_history[-1])
		# time.append(time_history[-1])
		# u.append(u_history[-1])


		print('----q ranges-----')
		for i in range(15):
			print(np.max(np.array(q)[:,i]),np.min(np.array(q)[:,i]))
		print('----q_dot rabges-----')
		for i in range(15):
			print(np.max(np.array(q_dot)[:,i]),np.min(np.array(q_dot)[:,i]))

		print('----u ranges ------')
		for i in range(12):
			print(np.max(np.array(u)[:,i]),np.min(np.array(u)[:,i]))

		#print(time,q[0][0],q[-1][0])
		

		np.save('results/PID_trajectory/11/q_init_guess.npy',np.array(q))
		np.save('results/PID_trajectory/11/q_dot_init_guess.npy',np.array(q_dot))
		np.save('results/PID_trajectory/11/u_init_guess.npy',np.array(u))
		np.save('results/PID_trajectory/11/time_init_guess.npy',np.array(time))


		#print(len(q_history),len(u_history))

		#print(np.shape(q_history))
		# plt.plot(time_history,q_history[:,4])
		# plt.show()



def trajectory_loader2():
	"""
	Takes a trajectory whose state does not match the control exactly and make it so by playing the control

	"""
		
	#1st second is standing still, the rest of 8 second is walking (1,3 swing first, then 2,4 swing)
	dt = 0.05
	x0 = np.concatenate((np.load('results/PID_trajectory/4/q_init_guess.npy')[0],np.load('results/PID_trajectory/4/q_dot_init_guess.npy')[0]))
	u = np.load('results/PID_trajectory/4/u_init_guess.npy')
	#x = [x0.tolist()]

	# x0 = np.load('results/PID_trajectory/3/x_init_guess.npy')[0]
	# u = np.load('results/PID_trajectory/3/u_init_guess.npy')

	q_2D = np.array([0]*15)[np.newaxis].T
	q_dot_2D = np.array([0.0]*15)[np.newaxis].T
	simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = dt, dyn = 'diffne',print_level = 0,augmented = True,extrapolation = True, integrate_dt = dt)
	simulator.reset(x0[0:15],x0[15:30])

	#vis robot
	robot = robosimian()
	world = robot.get_world()
	robot.set_q_2D_(x0[0:15])
	robot.set_q_dot_2D_(x0[15:30])

	vis.add("world",world)
	vis.show()
	vis.addText('time','time: '+str(0))
	time.sleep(5.0)
	simulation_time = 0.0

	x_history = []
	time_history = []

	counter = 0

	while vis.shown() and (simulation_time < 9.001):
		
		x_history.append(simulator.getConfig().tolist() + simulator.getVel().tolist())
		time_history.append(simulation_time)
		vis.lock()
		robot.set_q_2D_(simulator.getConfig())
		tau = u[counter]
		vis.clearText()
		vis.addText('time','time: '+str(simulation_time))
		simulator.simulateOnce(tau,continuous_simulation = True)

		print(tau)
		simulation_time += dt
		vis.unlock()
		time.sleep(dt)
		counter += 1
	np.save('results/PID_trajectory/7/x_init_guess.npy',np.array(x_history))
	np.save('results/PID_trajectory/7/u_init_guess.npy',u)
	np.save('results/PID_trajectory/7/time_init_guess.npy',np.array(time_history))


	print(np.shape(x_history))
if __name__=="__main__":
	trajectory_loader(start = 1.0,end = 10.0,dt = 0.05, number = 2)
	#trajectory_loader2()