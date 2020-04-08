import math
import numpy as np
from robosimian_GM_simulation import robosimianSimulator
from copy import deepcopy
#global robot
q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T
robot = robosimianSimulator(q_2D = q_2D,q_dot_2D = q_dot_2D,dt = 0.005,formulation = 'V')
def jac_dyn(x, u, p=None):
	a,J_SA = robot.getDynJac(x,u)
	print('a:',a)
	a = np.concatenate([x[15:30],a])		

	#Forward Finite Difference to get the 
	# eps = 1e-4
	# J = np.zeros((30,1+30+12))
	# for i in range(1):
	# 	#i = 1
	# 	FD_vector = np.zeros(30)
	# 	FD_vector[i] = eps
	# 	print('x is:',x)
	# 	tmp_x = np.add(x,FD_vector)
	# 	print('tmp_x is:',tmp_x)
	# 	tmp_a,unused = robot.getDynJac(tmp_x,u)
	# 	J[:,i+1] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],tmp_a]),a),1.0/eps)
	# 	C1 = deepcopy(J[:,i+1])
	# 	#print('SA:',J_SA[:,i])
	eps = 1e-7
	J = np.zeros((30,1+30+12))

	print('----')

	for i in [0,1,2,3]:
		FD_vector = np.zeros(30)
		FD_vector[i] = eps
		tmp_x = np.add(x,FD_vector)
		tmp_a,unused = robot.getDynJac(tmp_x,u)
		print('tmp_a',tmp_a)

		J[:,i+1] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],tmp_a]),a),1.0/eps)
		#print('SA:',J_SA[:,i])
	# eps = 1e-7
	# J = np.zeros((30,1+30+12))
	# for i in range(1):
	# 	#i = 1
	# 	FD_vector = np.zeros(30)
	# 	FD_vector[i] = eps
	# 	tmp_x = np.add(x,FD_vector)
	# 	tmp_a,unused = robot.getDynJac(tmp_x,u)
	# 	J[:,i+1] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],tmp_a]),a),1.0/eps)
	# 	C3 = J[:,i+1]


	print(J[:,1:4])
	print('SA',J_SA[:,0:3])




	# for i in range(12):
	# 	FD_vector = np.zeros(12)
	# 	FD_vector[i] = eps
	# 	tmp_u = np.add(u,FD_vector)
	# 	tmp_a = self.robot.getDynamics(x,tmp_u)
	# 	J[:,i+1+30] = np.multiply(np.subtract(np.concatenate([x[15:30],tmp_a]),a),1.0/eps)
	# print("Jac done")
	#maybe convert J to coomatrix
	return a,J

x0 = np.array([0,0.943,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15) #0.936 -- -0.08 ankle depth
u0 = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])
jac_dyn(x0,u0)