import math
import numpy as np
from robosimian_GM_simulation import robosimianSimulator
from copy import deepcopy
import copy
import configs
#global robot
# q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
# 	+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
# q_dot_2D = np.array([0.0]*15)[np.newaxis].T
# robot = robosimianSimulator(q = q_2D,q_dot= q_dot_2D,dt = 0.005,solver = 'cvxpy',print_level = 1)
# def jac_dyn(x, u, p=None):

# 	#calculate accleration and dynamics jacobian
# 	a,J_SA = robot.getDynJac(x,u)
# 	# print(x,np.ravel(a))
# 	a = np.concatenate([x[15:30],np.ravel(a)])		

# 	#calculate jacobian with finite-difference
# 	eps = 1e-6
# 	J = np.zeros((30,30+12))
# 	for i in [0,1,2,3]:
# 		FD_vector = np.zeros(30)
# 		FD_vector[i] = eps
# 		tmp_x = np.add(x,FD_vector)
# 		tmp_a,_,_,_= robot.getDyn(tmp_x,u)
# 		J[:,i] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],np.ravel(tmp_a)]),a),1.0/eps)

# 	print('FD')
# 	print(J[15:30,0:4])
# 	#print(J[15:30,4:8])
# 	print('SA')
# 	print(J_SA[15:30,0:4])
# 	#print(J_SA[15:30,4:8])

# 	return
# x0 = np.array(configs.q_staggered_limbs+[0.0]*15) #0.936 -- -0.08 ankle depth
# x0[1] = 1.0
# u0 = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
# 	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])

# jac_dyn(x0,u0)

####
class checker:
	def __init__(self,N = 1):
		q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
			+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
		q_dot_2D = np.array([0.0]*15)[np.newaxis].T
		self.dt = 0.01
		self.robot = robosimianSimulator(q = q_2D,q_dot= q_dot_2D,dt = self.dt,solver = 'cvxpy',print_level = 0)
		self.N = N

	def jac_dyn(self,x, u, p=None):

		current_x = copy.deepcopy(x)
		u0 = copy.deepcopy(u) #this remains the same
		#simualte 10 times at 100Hz for an interval of 0.1s, using the same control
		total_f = np.zeros(30)
		total_J = np.zeros((30,42))
		K = np.eye(42) #this is dXidX0
		for i in range(self.N):
			old_x = copy.deepcopy(current_x)
			a,J_tmp,current_x = self.robot.getDynJac(current_x,u0,True) #This is a numpy column 2D vector N*1
			total_f = total_f + np.concatenate([old_x[15:30],a.ravel()])
			if i < self.N - 1:			
				total_J += J_tmp@K
				dxidxi_1 = np.zeros((42,42))
				dxidxi_1[0:15,0:15] = np.eye(15)+self.dt*J_tmp[0:15,0:15]+(self.dt**2)*J_tmp[15:30,0:15]
				dxidxi_1[0:15,15:30] = self.dt*np.eye(15)+(self.dt**2)*J_tmp[15:30,15:30]
				dxidxi_1[0:15,30:42] = (self.dt**2)*J_tmp[15:30,30:42]
				dxidxi_1[15:30,0:15] = J_tmp[0:15,0:15] + self.dt*J_tmp[15:30,0:15]
				dxidxi_1[15:30,15:30] = np.eye(15) + self.dt*J_tmp[15:30,15:30]
				dxidxi_1[15:30,30:42] = self.dt*J_tmp[15:30,30:42]
				dxidxi_1[30:42,30:42] = np.eye(12)
				K = dxidxi_1@K
			else:
				total_J += J_tmp@K
		J = np.zeros((30,30+12))
		J[:,0:42] = total_J/self.N

		return total_f/self.N,J

	def dyn(self,x,u,p=None):
		current_x = copy.deepcopy(x)
		u0 = copy.deepcopy(u) #this remains the same

		#simualte 10 times at 100Hz for an interval of 0.1s, using the same control
		total_f = np.zeros(30)
		for i in range(self.N):
			old_x = copy.deepcopy(current_x)
			a,current_x = self.robot.getDyn(current_x,u0,True) #This is a numpy column 2D vector N*1
			total_f = total_f + np.concatenate([old_x[15:30],a.ravel()])

		#print(total_f/10.0)
		return total_f/self.N #1D numpy array

x = np.array(configs.q_staggered_limbs+[0.0]*15) #0.936 -- -0.08 ankle depth
x[1] = 1.0
u = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])

checker = checker(8)
a,J_SA = checker.jac_dyn(x, u)
#print(a)
#calculate jacobian with finite-difference
eps = 1e-6
J = np.zeros((30,30+12))
for i in [0,1,2,3]:
	FD_vector = np.zeros(30)
	FD_vector[i] = eps
	tmp_x = np.add(x,FD_vector)
	tmp_a = checker.dyn(tmp_x,u)
	J[:,i] = np.multiply(np.subtract(tmp_a,a),1.0/eps)

print('FD')
print(J[15:30,0:4])
#print(J[15:30,4:8])
print('SA')
print(J_SA[15:30,0:4])
#print(J_SA[15:30,4:8])

