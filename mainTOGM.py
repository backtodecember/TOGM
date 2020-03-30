#main file planning trajectories
#using trajoptlib
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import logging
from trajoptlib.io import get_onoff_args
from trajoptlib import System, NonLinearPointObj, LqrObj
from trajoptlib import TrajOptProblem
from trajoptlib import OptConfig, OptSolver
from trajoptlib.utility import show_sol
from scipy.sparse import coo_matrix
import math
from robosimian_GM_simulation import robosimianSimulator
class Robosimian(System):
	def __init__(self):
		System.__init__(self,nx=30,nu=12,np=0,ode='Euler')
		q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	 		+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
		q_dot_2D = np.array([0.0]*15)[np.newaxis].T
		self.robot = robosimianSimulator(q_2D = q_2D,q_dot_2D = q_dot_2D,dt = 0.005,formulation = 'V')

	def dyn(self,t,x,u,p=None):  
		a = self.robot.getDynamics(x,u) #This is a numpy column 2D vector N*1
		return np.concatenate([x[15:30],a]) #1D numpy array


	def jac_dyn(self, t, x, u, p=None):
		a,J_tmp = self.robot.getDynJac(x,u)
		a = np.concatenate([x[15:30],a])		
		J = np.zeros((30,1+30+12))
		J[:,1:43] = J_tmp
		return a,J

	####older version that uses naive FD for dynamics
	# def jac_dyn(self, t, x, u, p=None):
	# 	a = self.robot.getDynamics(x,u)
	# 	a = np.concatenate([x[15:30],a])		

	# 	#Forward Finite Difference to get the 
	# 	eps = 1e-4
	# 	J = np.zeros((30,1+30+12))
	# 	for i in range(30):
	# 		FD_vector = np.zeros(30)
	# 		FD_vector[i] = eps
	# 		tmp_x = np.add(x,FD_vector)
	# 		tmp_a = self.robot.getDynamics(tmp_x,u)
	# 		J[:,i+1] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],tmp_a]),a),1.0/eps)
	# 	for i in range(12):
	# 		FD_vector = np.zeros(12)
	# 		FD_vector[i] = eps
	# 		tmp_u = np.add(u,FD_vector)
	# 		tmp_a = self.robot.getDynamics(x,tmp_u)
	# 		J[:,i+1+30] = np.multiply(np.subtract(np.concatenate([x[15:30],tmp_a]),a),1.0/eps)
	# 	print("Jac done")
	# 	#maybe convert J to coomatrix
	# 	return a,J


sys = Robosimian()
N = 41
t0 = 0.0
tf = 0.2
#This uses direct transcription
problem = TrajOptProblem(sys,N,t0,tf,gradmode = True)
# penetrationConstr = penetrationConstraint(30)
# problem.add_constr(penetrationConstr,path = True)
R = np.array([1.0]*12)
cost = LqrObj(R = R)
problem.xbd = [np.array([-0.5,0.85,-1,-1.57-0.7,-0.7,-0.7,1.57-0.7,-0.7,-0.7,-1.57-0.7,-0.7,-0.7,1.57-0.7,-0.7,-0.7] + [-2]*15),\
	np.array([0.5,1.2,1,-1.57+0.7,0.7,0.7,1.57+0.7,0.7,0.7,-1.57+0.7,0.7,0.7,1.57+0.7,0.7,0.7] + [2]*15)]
problem.ubd = [np.array([-1000]*12),np.array([1000]*12)]
problem.x0bd = [np.array([0,0.936,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15),\
	np.array([0,0.936,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15)] 
#problem.xfbd = [np.array([1.0,0.936,0]+[-10.0]*12 + [0.0]*15),np.array([1.0,0.936,0]+[10.0]*12 +[0.0]*15)] ## set the goal here 
problem.xfbd = [np.array([0,0.936,0]+[-10.0]*12 + [0.0]*15),np.array([0.0,0.936,0]+[10.0]*12 +[0.0]*15)] #just stay in place...


problem.add_lqr_obj(cost)
problem.preProcess()
#cfg = OptConfig(backend='snopt', deriv_check=1, print_file='tmp.out')
cfg = OptConfig(backend = 'ipopt', deriv_check=0, print_level = 1,print_file='tmp.out')
slv = OptSolver(problem, cfg)
startTime = time.time()

single_traj_guess = [0,0.936,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15
single_u_guess = [6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186]
traj_guess = []
u_guess = []
for i in range(N):
	traj_guess.append(single_traj_guess)
	u_guess.append(single_u_guess)

guess = problem.genGuessFromTraj(X= np.array(traj_guess), U=np.array(u_guess), t0 = 0, tf = tf)
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

np.save('solution_u',sol['u'])
np.save('solution_x',sol['x'])
# if rst.flag == 1:
# 	print(rst.flag)
# 	sol = problem.parse_sol(rst.sol.copy())
# 	show_sol(sol)