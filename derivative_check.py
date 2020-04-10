import math
import numpy as np
from robosimian_GM_simulation import robosimianSimulator
from copy import deepcopy
#global robot
q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T
robot = robosimianSimulator(q = q_2D,q_dot= q_dot_2D,dt = 0.005,solver = 'cvxpy')
def jac_dyn(x, u, p=None):

	#calculate accleration and dynamics jacobian
	a,J_SA = robot.getDynJac(x,u)
	print(x,np.ravel(a))
	a = np.concatenate([x[15:30],np.ravel(a)])		

	#calculate jacobian with finite-difference
	eps = 1e-4
	J = np.zeros((30,1+30+12))
	for i in [0,1,2,3,4,5]:
		FD_vector = np.zeros(30)
		FD_vector[i] = eps
		tmp_x = np.add(x,FD_vector)
		tmp_a= robot.getDyn(tmp_x,u)
		J[:,i+1] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],np.ravel(tmp_a)]),a),1.0/eps)

	print('FD')
	print(J[15:30,1:5])
	print(J[15:30,5:9])
	print('SA')
	print(J_SA[15:30,0:4])
	print(J_SA[15:30,4:8])

	return

x0 = np.array([0,0.936,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15) #0.936 -- -0.08 ankle depth
u0 = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])

# x0 = np.array([0.0,0.936,0.0] + [0.6- 1.5708,0.0,-0.6]+[-0.6+1.5708,0.0,0.6]+[0.6-1.5708,0.0,-0.6] \
# 	+[-0.6+1.5708,0.0,0.6]+[0.0]*15)
# u0 = np.array([6.08309021,0.81523653, 2.53641154 ,-5.50487329, -0.54710471,-2.57836468,\
# 	5.50487329, 0.54710471,2.57836468, -6.08309021,-0.81523653, -2.53641154])
#u0 = np.zeros(12)


jac_dyn(x0,u0)