import numpy as np
from klampt.io import loader

# trajectory = loader.loadTrajectory('data/trotting_gait')
# milestones = trajectory.milestones
# print(len(milestones))
# #print(trajectory.times)
# total_time = trajectory.endTime()
# print(total_time)
# print(milestones[0])




# x = np.load('results/PID_trajectory/3/x_init_guess.npy')
# u = np.load('results/PID_trajectory/3/u_init_guess.npy')
# t = np.load('results/PID_trajectory/3/time_init_guess.npy')

# print('time list length',np.shape(t))
# print('x[0]:',x[0])
# print('x[-1]:',x[-1])
# print('u[0]:',u[0])
# print('t[0]',t[0])
# print('t[-1]:',t[-1])

#traj_guess = np.hstack((np.load('results/PID_trajectory/2/q_init_guess.npy'),np.load('results/PID_trajectory/2/q_dot_init_guess.npy')))
# u_guess = np.load('results/PID_trajectory/2/u_init_guess.npy')
# x_guess = np.load('results/PID_trajectory/2/x_init_guess.npy')

# print(np.shape(u_guess),np.shape(x_guess))

# np.save('results/PID_trajectory/2/x_init_guess.npy',x)
# np.save('results/PID_trajectory/2/u_init_guess.npy',u)

# a = [1,2]
# b = np.array(a)
# print(type(a),type(b))
# if isinstance(a,list):
#     print('list')
# if isinstance(b,np.ndarray):
#     print('numpy array')    

# (m,) = np.shape(a)
# N = len(a)
# print(m)
# print(N)

# x = np.hstack((np.load('results/PID_trajectory/4/q_init_guess.npy'),np.load('results/PID_trajectory/4/q_dot_init_guess.npy')))
# # u = np.load('results/PID_trajectory/4/u_history.npy')
# # print(np.shape(x),np.shape(u))
# Q = np.zeros((30,30))
# Q[1,1] = 0.01
# Q[15,15] = 10.0
# xbase = np.zeros(30)
# xbase[1] = 0.9
# xbase[15] = 0.2
# total = 0.0
# for i in range(180):
#     v = x[i,:][np.newaxis]
#     #print(np.shape(v))
#     val = v@Q@v.T
#     total += val[0,0]
#     print(val[0,0])
# print(total)

x = np.load('results/21/run3/solution_x500.npy')
print(x[:,15])