#main file planning trajectories
#using trajoptlib
#For a trajectory that is longer, e.g. 10s, it involves too many nodes if we control at 100Hz
# this file tries to do it at 10Hz, but to enforce correctness, the underlying dynamics and kinematics
# is still enforced at 100Hz.
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import logging
from trajoptlib.io import get_onoff_args
from trajoptlib import System, NonLinearPointObj, LqrObj
from trajoptlib import LinearConstr,NonLinearObj
from trajoptlib import TrajOptProblem,TrajOptCollocProblem
from trajoptlib import OptConfig, OptSolver
from trajoptlib.utility import show_sol
from scipy.sparse import coo_matrix
import math
from robosimian_GM_simulation import robosimianSimulator
import pdb
import configs
from klampt.math import vectorops as vo
import copy
class Robosimian(System):
	def __init__(self):
		System.__init__(self,nx=30,nu=12,np=0,ode='Euler')
		q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	 		+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
		q_dot_2D = np.array([0.0]*15)[np.newaxis].T
		self.dt = 0.01
		self.robot = robosimianSimulator(q= q_2D,q_dot = q_dot_2D,dt = self.dt,solver = 'cvxpy', augmented = True)
		self.dudx = np.zeros((12,42))
		for i in range(12):
			self.dudx[i,i+30] = 1.0
		self.N = 8

	def dyn(self,t,x,u,p=None):
		current_x = copy.deepcopy(x)
		u0 = copy.deepcopy(u) #this remains the same

		#simualte 10 times at 100Hz for an interval of 0.1s, using the same control
		total_f = np.zeros(30)
		for i in range(self.N):
			old_x = copy.deepcopy(current_x)
			a,current_x = self.robot.getDyn(current_x,u,True) #This is a numpy column 2D vector N*1
			total_f = total_f + np.concatenate([old_x[15:30],a.ravel()])

		return total_f/self.N #1D numpy array


	def jac_dyn(self, t, x, u, p=None):
		current_x = copy.deepcopy(x)
		u0 = copy.deepcopy(u) #this remains the same
		#simualte 10 times at 100Hz for an interval of 0.1s, using the same control
		total_f = np.zeros(30)
		total_J = np.zeros((30,42))
		K = np.eye(42) #this is dXidX0
		for i in range(self.N):
			old_x = copy.deepcopy(current_x)
			a,J_tmp,current_x = self.robot.getDynJac(current_x,u,True) #This is a numpy column 2D vector N*1
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
		J = np.zeros((30,1+30+12))
		J[:,1:43] = total_J/self.N

		#print(total_f/10.0)
		#print(J)
		return total_f/self.N,J

#These are the force and torque that could support the robot in place
single_traj_guess = [0,0.915,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15
single_u_guess = configs.u_augmented_mosek


sys = Robosimian()
global N
N = 114 # 8Hz about 9s in total
t0 = 0.0
tf = 0.08*(N-1)
#This uses direct transcription
#problem = TrajOptProblem(sys,N,t0,tf,gradmode = True)
problem = TrajOptProblem(sys,N,t0,tf,gradmode = True)

#####setting 8&9&10
# Q = np.array([0]*15 + [1.0]*15)
# cost = LqrObj(Q = Q)#,ubase = np.array(single_u_guess))
# diff = [0.3]*15
# problem.xbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]
# problem.ubd = [np.array([-1000]*12),np.array([1000]*12)]
# problem.x0bd = [np.array(single_traj_guess),np.array(single_traj_guess)]
# problem.xfbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]

#####This is for setting 12
problem.xbd = [np.array([-0.2,0.4,-0.3] + [-math.pi]*12 + [-3]*15),np.array([5,1.1,0.3] + [math.pi]*12 + [3]*15)]
problem.ubd = [np.array([-300]*12),np.array([300]*12)]
problem.x0bd = [np.array([-0.1,0.4,-0.3] + [-math.pi]*12 + [-3]*15),np.array([0.1,1.1,0.3] + [math.pi]*12 + [3]*15)]
problem.xfbd = [np.array([0.2,0.4,-0.3] + [-math.pi]*12 + [-3]*15),np.array([5,1.1,0.3] + [math.pi]*12 + [3]*15)]

class cyclicConstr(LinearConstr):
	def __init__(self):
		global N
		dmin = 0.4
		dmax = 2.0
		nX = 30
		nU = 12
		nP = 0
		A = np.zeros((nX,N*(nX+nU+nP)))
		lb = np.zeros(nX)
		ub = np.zeros(nX)
		#enough x translation
		A[0,0] = -1.0
		A[0,0+(N-1)*(nX+nU+nP)] = 1.0
		lb[0] = dmin
		ub[0] = dmax
		#Remain same for the other state dimensions 0.4
		for i in range(nX):
			if i > 0:
				A[i,i] = 1.0
				A[i,i+(N-1)*(nX+nU+nP)] = -1.0
		LinearConstr.__init__(self,A,lb = lb,ub = ub)

class transportationCost(NonLinearObj):
	def __init__(self):
		self.nX = 30
		self.nU = 12
		self.nP = 0
		global N
		self.N = N
		self.first_q_dot = 18
		self.NofJoints = 12
		self.first_u = 30
		NonLinearObj.__init__(self,nsol = self.N*(self.nX+self.nU+self.nP),nG = self.N*(self.nX+self.nU)-self.N*self.first_q_dot +2  )

		#print(self.N*(self.nX+self.nU)-self.N*self.first_q_dot +2)
	def __callg__(self,x, F, G, row, col, rec, needg):
		effort_sum = 0.0
		for i in range(self.N):
			for j in range(self.NofJoints):
				#q_dot * u
				effor_sum = effort_sum + (x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
					x[i*(self.nX+self.nU+self.nP)+self.first_u+j])**2
		F[:] = effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0])
		if needg:
			Gs = []
			nonzeros = [0]
			# for i in range(self.N):
			# 	for j in range(self.first_q_dot):
			# 		G[i*(self.nX+self.nU+self.nP)+j] = 0.0
			Gs.append(effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0])**2)
			#G[0] = effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0])**2
			#G[(self.N-1)*(self.nX+self.nU+self.nP)] = -effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0])**2
			d = x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]
			for i in range(self.N-1):
				for j in range(self.NofJoints):
					# G[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j] = 2.0*x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
					# 	x[i*(self.nX+self.nU+self.nP)+self.first_u+j]**2
					Gs.append(2.0/d*x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j]**2)
					nonzeros.append(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j)
				for j in range(self.NofJoints):
					# G[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j+self.NofJoints] = 2.0*x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]**2*\
					# 	x[i*(self.nX+self.nU+self.nP)+self.first_u+j]
					Gs.append(2.0/d*(x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]**2)*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j])
					nonzeros.append(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j+self.NofJoints)
			Gs.append(-effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0])**2)
			nonzeros.append((self.N-1)*(self.nX+self.nU+self.nP))
			for i in [self.N-1]:
				for j in range(self.NofJoints):
					Gs.append(2.0/d*x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j]**2)
					nonzeros.append(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j)
				for j in range(self.NofJoints):
					Gs.append(2.0/d*(x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]**2)*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j])
					nonzeros.append(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j+self.NofJoints)
			G[:] = Gs
			if rec:
				row[:] = [0]
				col[:] = nonzeros

#settings 7-11
#problem.add_lqr_obj(cost)
#setting 12
c = transportationCost()
constr1 = cyclicConstr()
problem.addNonLinearObj(c)
problem.addLinearConstr(constr1)

startTime = time.time()
problem.preProcess()
print('preProcess took:',time.time() - startTime)
#print_level 1 for SNOPT is verbose enough
#print_level 0-12 for IPOPT 5 is appropriate
cfg = OptConfig(backend='ipopt', print_file='temp_files/tmp.out', print_level = 5, opt_tol = 1e-4, fea_tol = 1e-4)
slv = OptSolver(problem, cfg)

#setting 12
# slv.solver.setWorkspace(4000000,5000000)
# print('set workspace done')

startTime = time.time()


#setting 1-13
# traj_guess1 = []
# u_guess1 = []
# for i in range(N):
# 	tmp = copy.copy(single_traj_guess)
# 	#settting 12
# 	#tmp[0] = i*0.002
# 	traj_guess1.append(tmp)
# 	u_guess1.append(single_u_guess)
#print(np.shape(np.array(traj_guess1)))
# guess = problem.genGuessFromTraj(X= np.array(traj_guess), U=np.array(u_guess), t0 = 0, tf = tf)

#setting 14
traj_guess = np.hstack((np.load('results/PID_trajectory/1/q_init_guess.npy'),np.load('results/PID_trajectory/1/q_dot_init_guess.npy')))
u_guess = np.load('results/PID_trajectory/1/u_init_guess.npy')
#print(np.shape(np.array(traj_guess)))
guess = problem.genGuessFromTraj(X= traj_guess, U= u_guess, t0 = 0, tf = tf)

result = problem.parseF(guess)
rst = slv.solve_guess(guess)

print('Took', time.time() - startTime)
print("========results=======")
print(rst.flag)
print(rst.fval,np.shape(rst.fval))
print(rst.sol,np.shape(rst.sol))
print(np.shape(rst.lmd))

sol = problem.parse_sol(rst.sol.copy())
np.save('temp_files/solution_u',sol['u'])
np.save('temp_files/solution_x',sol['x'])
