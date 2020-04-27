#main file planning trajectories
#using trajoptlib
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import logging
from trajoptlib.io import get_onoff_args
from trajoptlib import System, NonLinearPointObj, LqrObj,linearConstr,nonLinearObj
from trajoptlib import TrajOptProblem
from trajoptlib import OptConfig, OptSolver
from trajoptlib.utility import show_sol
from scipy.sparse import coo_matrix
import math
from robosimian_GM_simulation import robosimianSimulator
import pdb
import configs
from klampt.math import vectorops as vo
from trajoptlib import 
class Robosimian(System):
	def __init__(self):
		System.__init__(self,nx=30,nu=12,np=0,ode='Euler')
		q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	 		+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
		q_dot_2D = np.array([0.0]*15)[np.newaxis].T
		self.robot = robosimianSimulator(q= q_2D,q_dot = q_dot_2D,dt = 0.005,solver = 'cvxpy', augmented = True)

	def dyn(self,t,x,u,p=None):  
		a = self.robot.getDyn(x,u) #This is a numpy column 2D vector N*1
		return np.concatenate([x[15:30],a]) #1D numpy array


	def jac_dyn(self, t, x, u, p=None):

		a,J_tmp = self.robot.getDynJac(x,u)
		a = np.concatenate([x[15:30],np.ravel(a)])		
		J = np.zeros((30,1+30+12))
		J[:,1:43] = J_tmp
		return a,J

	###older version that uses naive FD for dynamics
	# def jac_dyn(self, t, x, u, p=None):
	# 	a = np.ravel(self.robot.getDyn(x,u))
	# 	a = np.concatenate([x[15:30],a])		

	# 	#Forward Finite Difference to get the 
	# 	eps = 1e-4
	# 	J = np.zeros((30,1+30+12))
	# 	for i in range(30):
	# 		FD_vector = np.zeros(30)
	# 		FD_vector[i] = eps
	# 		tmp_x = np.add(x,FD_vector)
	# 		tmp_a = self.robot.getDyn(tmp_x,u)
	# 		J[:,i+1] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],np.ravel(tmp_a)]),a),1.0/eps)
	# 	for i in range(12):
	# 		FD_vector = np.zeros(12)
	# 		FD_vector[i] = eps
	# 		tmp_u = np.add(u,FD_vector)
	# 		tmp_a = self.robot.getDyn(x,tmp_u)
	# 		J[:,i+1+30] = np.multiply(np.subtract(np.concatenate([x[15:30],np.ravel(tmp_a)]),a),1.0/eps)
	# 	#print("Jac done")
	# 	#maybe convert J to coomatrix
	# 	return a,J

class cyclicConstr(linearConstr):
	def __init__(self):
		linearPointConstr.__init__()

class transportationCost(nonLinearObj):
	def __init__(self):
		nonLinearObj.__init__()

#These are the force and torque that could support the robot in place.
single_traj_guess = [0,0.915,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15
single_u_guess = configs.u_augmented_mosek


sys = Robosimian()
N = 50
t0 = 0.0
tf = 0.25
#This uses direct transcription
problem = TrajOptProblem(sys,N,t0,tf,gradmode = True)
# penetrationConstr = penetrationConstraint(30)
# problem.add_constr(penetrationConstr,path = True)


#### different setttings start from here
#####setting 7
# R = np.array([1.0]*12)
# cost = LqrObj(R = R)#,ubase = np.array(single_u_guess))
# diff = [0.3]*15
# problem.xbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]
# problem.ubd = [np.array([-1000]*12),np.array([1000]*12)]
# problem.x0bd = [np.array(single_traj_guess),np.array(single_traj_guess)]
# problem.xfbd = [np.array(single_traj_guess),np.array(single_traj_guess)] #just stay in place...

#####setting 8&9&10
# Q = np.array([0]*15 + [1.0]*15)
# cost = LqrObj(Q = Q)#,ubase = np.array(single_u_guess))
# diff = [0.3]*15
# problem.xbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]
# problem.ubd = [np.array([-1000]*12),np.array([1000]*12)]
# problem.x0bd = [np.array(single_traj_guess),np.array(single_traj_guess)]
# problem.xfbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]

#####setting 11
# Here we have the toros basically shifting forward
# x_base = np.array([0.0]*15 + [0.1] + [0.0]*14)
# Q = np.array([0.0]*15 + [10.0]*3 + [0.01]*12)
# cost = LqrObj(Q = Q,xbase = x_base)#,ubase = np.array(single_u_guess))
# diff = [2]*15
# #be lax on the x bounds
# problem.xbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]
# problem.ubd = [np.array([-100]*12),np.array([100]*12)]

# x0lower = vo.sub(single_traj_guess,[0]+[0.005]*14+[2]*15)
# x0upper = vo.add(single_traj_guess,[0]+[0.005]*14+[2]*15)
# problem.x0bd = [np.array(x0lower),np.array(x0upper)]
# problem.xfbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]

#####Original setting
# R = np.array([1.0]*12)
# cost = LqrObj(R = R)#,ubase = np.array(single_u_guess))
# problem.xbd = [np.array([-0.5,0.85,-1,-1.57-0.7,-0.7,-0.7,1.57-0.7,-0.7,-0.7,-1.57-0.7,-0.7,-0.7,1.57-0.7,-0.7,-0.7] + [-2]*15),\
# 	np.array([0.5,1.2,1,-1.57+1,0.7,0.7,1.57+1,0.7,0.7,-1.57+0.7,0.7,0.7,1.57+1,0.7,0.7] + [2]*15)]
# problem.ubd = [np.array([-1000]*12),np.array([1000]*12)]
# problem.x0bd = [np.array([0,0.936,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15),\
# 	np.array([0,0.936,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15)] 
# #problem.xfbd = [np.array([1.0,0.936,0]+[-10.0]*12 + [0.0]*15),np.array([1.0,0.936,0]+[10.0]*12 +[0.0]*15)] ## set the goal here 
# problem.xfbd = [np.array([0,0.936,0]+[-10.0]*12 + [0.0]*15),np.array([0.0,0.936,0]+[10.0]*12 +[0.0]*15)] #just stay in place...


#settings 7-11
#problem.add_lqr_obj(cost)

#setting 12
problem.add
problem.preProcess()
#print_level 1 for SNOPT is verbose enough
#print_level 0-12 for IPOPT 5 is appropriate
cfg = OptConfig(backend='snopt', print_file='temp_files/tmp.out', print_level = 1, opt_tol = 1e-4, fea_tol = 1e-4)
#cfg = OptConfig(backend = 'ipopt',print_level = 5,print_file='tmp.out',opt_tol = 1e-4)
slv = OptSolver(problem, cfg)
startTime = time.time()



traj_guess = []
u_guess = []
for i in range(N):
	traj_guess.append(single_traj_guess)
	u_guess.append(single_u_guess)

guess = problem.genGuessFromTraj(X= np.array(traj_guess), U=np.array(u_guess), t0 = 0, tf = tf)

result = problem.parseF(guess)
#print(result)
#pdb.set_trace()
#print(guess)
rst = slv.solve_guess(guess)

#rst = slv.solve_rand()
print('Took', time.time() - startTime)
print("========results=======")
print(rst.flag)
print(rst.fval,np.shape(rst.fval))
print(rst.sol,np.shape(rst.sol))
print(np.shape(rst.lmd))
sol = problem.parse_sol(rst.sol.copy())
print(sol)

np.save('temp_files/solution_u',sol['u'])
np.save('temp_files/solution_x',sol['x'])
# if rst.flag == 1:
# 	print(rst.flag)
# 	sol = problem.parse_sol(rst.sol.copy())
# 	show_sol(sol)