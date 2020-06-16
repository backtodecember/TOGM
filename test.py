import numpy as np
from klampt.io import loader

# trajectory = loader.loadTrajectory('data/trotting_gait')
# milestones = trajectory.milestones
# print(len(milestones))
# #print(trajectory.times)
# total_time = trajectory.endTime()
# print(total_time)
# print(milestones[0])




x = np.load('results/PID_trajectory/3/x_init_guess.npy')
u = np.load('results/PID_trajectory/3/u_init_guess.npy')
t = np.load('results/PID_trajectory/3/time_init_guess.npy')

print('time list length',np.shape(t))
print('x[0]:',x[0])
print('x[-1]:',x[-1])
print('u[0]:',u[0])
print('t[0]',t[0])
print('t[-1]:',t[-1])