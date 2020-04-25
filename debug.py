import math
import numpy as np
from copy import deepcopy
import configs
import matplotlib.pyplot as plt
from klampt.math import vectorops as vo

def initialize():
	from robosimian_GM_simulation import robosimianSimulator
	#global robot
	#these q's don't matter
	q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
		+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
	q_dot_2D = np.array([0.0]*15)[np.newaxis].T

	global robot
	robot = robosimianSimulator(q = q_2D,q_dot= q_dot_2D,dt = 0.005,solver = 'cvxpy',print_level = 1, augmented = True)

def jac_dyn(x, u ,eps = 1e-4):
	global robot
	#calculate accleration and dynamics jacobian
	a,J_SA = robot.getDynJac(x,u)
	#a,C,D,wc = robot.getDyn(x,u)
	#print('accleration:',a)
	print('------------------------------------------')
	a = np.concatenate([x[15:30],np.ravel(a)])		

	#calculate jacobian with finite-difference
	eps = 1e-5
	J = np.zeros((30,30+12))
	for i in [0,1,2,3,4]:#range(30):
		FD_vector = np.zeros(30)
		FD_vector[i] = eps
		tmp_x = np.add(x,FD_vector)
		tmp_a,_,_,_= robot.getDyn(tmp_x,u)
		#print('accleration:',tmp_a)
		J[:,i] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],np.ravel(tmp_a)]),a),1.0/eps)
		print('-----------------------',i,'-------------------')
	print('FD')
	print(J[15:30,0:4])
	# print(J[15:30,4:8])
	print('SA')
	print(J_SA[15:30,0:4])
	# print(J_SA[15:30,4:8])
	return

def jacobian(x,u):
	jac_dyn(x, u ,eps = 5e-4)
	jac_dyn(x, u ,eps = 1e-4)
	jac_dyn(x, u ,eps = 1e-5)


def compute(x,u):
	global robot
	#compute stuff
	#check how acceleration changes as the state changes
	dimension = 3

	x2_start = x[dimension]
	deltas = [1e-7,1e-6,5e-6,1e-5,5e-5,1e-4]
	#deltas = [1e-4]#,3.6e-6,3.7e-6]
	x_axis = []
	counter = 0
	for delta in deltas:
	    x[dimension] = x2_start + delta
	    a,_,_,_ = robot.getDyn(x,u)
	    if counter == 0:
	        As = a
	    else:
	        As = np.hstack((As,a))
	    x_axis.append(x[dimension] - x2_start)
	    counter = counter + 1
	np.save('temp_files/y_axis',As)
	np.save('temp_files/x_axis',x_axis)

def plot():
	As = np.load('temp_files/y_axis.npy')
	x_axis = np.load('temp_files/x_axis.npy')
	print(x_axis)
	plt.plot(np.log10(x_axis),As[0,:],np.log10(x_axis),As[1,:],np.log10(x_axis),As[2,:],np.log10(x_axis),As[3,:])
	plt.legend(['a1','a2','a3','a4'])
	plt.ylabel('acceleration')
	plt.xlabel('log(delta q4)')
	plt.title('Acceleration changes')
	plt.show()
if __name__=="__main__":
	x_new = vo.add(configs.q_staggered_limbs,[0.1,0.0,0.05,0.02,-0.03,-0.3,-0.1,0.15,-0.2,-0.3,0.25,-0.4,-0.07,0.2,-0.2])
	x0 = np.array(x_new+[0.0]*15) #0.936 -- -0.08 ankle depth
	x0 = np.array(configs.q_staggered_limbs+[0.0]*15) #0.936 -- -0.08 ankle depth
	x0[1] = 0.915
	x0 = x0 + np.random.rand(30)*0.1
	#x0[1] = 1.0
	u0 = np.array(configs.u_augmented) + np.array(np.random.rand(12))
	#u0 = np.array(np.random.rand(12))
	#u0 = np.zeros(12)
	#u0 = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	#	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])
	# x0 = np.array(configs.q_symmetric+[0.0]*15)
	# u0 = np.array([6.08309021,0.81523653, 2.53641154 ,-5.50487329, -0.54710471,-2.57836468,\
	#  	5.50487329, 0.54710471,2.57836468, -6.08309021,-0.81523653, -2.53641154])
	#u0 = np.zeros(12)


	initialize()
	compute(x0,u0)
	plot()

	#jac_dyn(x0,u0)
