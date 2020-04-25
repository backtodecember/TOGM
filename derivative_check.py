import math
import numpy as np
from robosimian_GM_simulation import robosimianSimulator
from copy import deepcopy
import configs
#global robot
q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T
robot = robosimianSimulator(q = q_2D,q_dot= q_dot_2D,dt = 0.005,solver = 'cvxpy',print_level = 1)
def jac_dyn(x, u, p=None):

	#calculate accleration and dynamics jacobian
	a,J_SA = robot.getDynJac(x,u)
	# print(x,np.ravel(a))
	a = np.concatenate([x[15:30],np.ravel(a)])		

	#calculate jacobian with finite-difference
	eps = 1e-6
	J = np.zeros((30,30+12))
	for i in [0,1,2,3]:
		FD_vector = np.zeros(30)
		FD_vector[i] = eps
		tmp_x = np.add(x,FD_vector)
		tmp_a,_,_,_= robot.getDyn(tmp_x,u)
		J[:,i] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],np.ravel(tmp_a)]),a),1.0/eps)

	print('FD')
	print(J[15:30,0:4])
	#print(J[15:30,4:8])
	print('SA')
	print(J_SA[15:30,0:4])
	#print(J_SA[15:30,4:8])

	return

x0 = np.array(configs.q_staggered_limbs+[0.0]*15) #0.936 -- -0.08 ankle depth
x0[1] = 1.0
#x0[2] = x0[2] + 0.01
u0 = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])


# x0 = np.array(configs.q_symmetric+[0.0]*15)
# x0[1] = 1.0
# u0 = np.array([6.08309021,0.81523653, 2.53641154 ,-5.50487329, -0.54710471,-2.57836468,\
# 	5.50487329, 0.54710471,2.57836468, -6.08309021,-0.81523653, -2.53641154])
#u0 = np.zeros(12)


jac_dyn(x0,u0)