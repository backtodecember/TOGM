import numpy as np
from robosimian_GM_simulation import robosimianSimulator
import matplotlib.pyplot as plt
q0 = np.array([0,0.936,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6])[np.newaxis].T
q_dot0 = np.zeros((15,1))
robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = 0.005, solver = 'cvxpy')

case = '4'
u = np.load('results/'+case+'/solution_u.npy')
u_traj = u
dt = 0.005
# for i in range(5):
# 	u_traj = np.vstack((u_traj,u))
x_traj = robot.simulateTraj(u_traj)

(m,n) = np.shape(x_traj)
time = np.array([i*dt for i in range(m)])

print(time)
for i in range(m):
	x_traj[i,:] = x_traj[i,:]-np.ravel(q0)

plt.plot(time,x_traj[:,0],time,x_traj[:,1],time,x_traj[:,2],time,x_traj[:,3],time,x_traj[:,4],time,x_traj[:,5])
plt.legend(['q1','q2','q3','q4','q5','q6'])
plt.ylabel('state')
plt.xlabel('time')
plt.title('trajectory')
plt.show()