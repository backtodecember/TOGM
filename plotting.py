import numpy as np
from robosimian_GM_simulation import robosimianSimulator
import matplotlib.pyplot as plt
import configs
from copy import deepcopy

q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
q_dot0 = np.zeros((15,1))
robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = 0.005, solver = 'cvxpy', augmented = True)

N = 100
case = '11-2'
u = np.load('results/'+case+'/solution_u.npy')
x_simulation = np.load('results/'+case+'/solution_x.npy')

u_traj = u

dt = 0.005



#x_traj = robot.simulateTraj(u_traj[0:N,:])
x_traj = x_simulation[:,0:15]
x_dot = x_simulation[:,15:30]



(m,n) = np.shape(x_traj)
time = np.array([i*dt for i in range(m)])

# print(time)
#offset = np.ravel(q0)
offset = deepcopy(x_traj[0,:])
for i in range(m):
	x_traj[i,:] = x_traj[i,:]-offset

## intergrate with static torque
# q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
# q_dot0 = np.zeros((15,1))
# robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = 0.005, solver = 'cvxpy', augmented = True)



# u_traj2 = np.array([configs.u_augmented_mosek]*N)
# x_traj2 = robot.simulateTraj(u_traj2[0:100,:])

# (m,n) = np.shape(x_traj2)
# time = np.array([i*dt for i in range(m)])

# # print(time)
# for i in range(m):
# 	x_traj2[i,:] = x_traj2[i,:]-np.ravel(q0)


# plt.plot(time,x_traj[:,0],'r',time,x_traj[:,1],'g',time,x_traj[:,2],'b',time,x_traj[:,3],'k',\
# 	time,x_traj2[:,0],'r:',time,x_traj2[:,1],'g:',time,x_traj2[:,2],'b:',time,x_traj2[:,3],'k:')
# plt.legend(['q1','q2','q3','q4','q1 static torque','q2','q3','q4'])
# plt.ylabel('state')
# plt.xlabel('time')
# plt.title('trajectory')
# plt.show()



#plot only 1 set
# plt.plot(time,x_traj[:,0],time,x_traj[:,1],time,x_traj[:,2],time,x_traj[:,3],time,x_traj[:,4],time,x_traj[:,5])
# plt.legend(['q1','q2','q3','q4','q5','q6'])
# plt.ylabel('state')
# plt.xlabel('time')
# plt.title('trajectory')
# plt.show()

#plot velocity
# plt.plot(time,x_dot[:,0],time,x_dot[:,1],time,x_dot[:,2],time,x_dot[:,3],time,x_dot[:,4],time,x_dot[:,5])
# plt.legend(['q1_dot','q2_dot','q3_dot','q4_dot','q5_dot','q6_dot'])
# plt.ylabel('state')
# plt.xlabel('time')
# plt.title('trajectory')
# plt.show()

#plot control

plt.plot(time,u[:,0],time,u[:,1],time,u[:,2],time,u[:,3],time,u[:,4],time,u[:,5])
plt.legend(['u1','u2','u3','u4','u5','u6'])
plt.ylabel('state')
plt.xlabel('time')
plt.title('controls')
plt.show()