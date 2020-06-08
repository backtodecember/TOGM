import numpy as np
import matplotlib.pyplot as plt
from klampt.math import vectorops as vo
def trajectory_loader(start,end,dt,number):
	"""
	load a trajectory into desired start and end, with  desired dt

	"""
	if number == 1:
		#1st second is standing still, the rest of 8 second is walking (1,3 swing first, then 2,4 swing)
		original_dt = 0.005
		start_walking  = 1.0
		end_walking = 10.0
		scale = round(dt/original_dt)
		path = "results/PID_trajectory/1/"
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

		q.append(q_history[-1])
		q_dot.append(q_dot_history[-1])
		time.append(time_history[-1])
		u.append(u_history[-1])
		print('----q-----')
		for i in range(15):
			print(np.max(np.array(q)[:,i]),np.min(np.array(q)[:,i]))
		print('----q_dot-----')
		for i in range(15):
			print(np.max(np.array(q_dot)[:,i]),np.min(np.array(q_dot)[:,i]))
		for i in range(12):
			print(np.max(np.array(u)[:,i]),np.min(np.array(u)[:,i]))

		#print(time,q[0][0],q[-1][0])
		

		# np.save('results/PID_trajectory/1/q_init_guess.npy',np.array(q))
		# np.save('results/PID_trajectory/1/q_dot_init_guess.npy',np.array(q_dot))
		# np.save('results/PID_trajectory/1/u_init_guess.npy',np.array(u))
		# np.save('results/PID_trajectory/1/time_init_guess.npy',np.array(time))



		#print(len(q_history),len(u_history))

		#print(np.shape(q_history))
		# plt.plot(time_history,q_history[:,4])
		# plt.show()

if __name__=="__main__":
	trajectory_loader(1.0,10.0,0.08,1)
	#print(round(0.08/0.01))