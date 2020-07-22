#main file planning trajectories
#using trajoptlib
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import logging
from trajoptlib.io import get_onoff_args
from trajoptlib import System, NonLinearPointObj, LqrObj,LinearObj
from trajoptlib import LinearConstr,NonLinearObj,NonLinearPointConstr
from trajoptlib import TrajOptProblem
from trajoptlib import OptConfig, OptSolver
from trajoptlib.utility import show_sol
from scipy.sparse import coo_matrix
import math
from robosimian_GM_simulation import robosimianSimulator
from robosimian_wrapper import robosimian
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
		dt = 0.05
		#Test 14+ should have extraploation set to be True
		self.robot = robosimianSimulator(q= q_2D,q_dot = q_dot_2D,dt = dt,dyn = 'diffne', augmented = True, extrapolation = True, integrate_dt = dt)

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

############################
#Initialize the problem    #
############################


##setting 1-21

sys = Robosimian()
global N
N = 181 #9s trajectory 0.05s
t0 = 0.0
tf = 0.05*(N-1)


##setting 22
# sys = Robosimian()
# global N
# N = 1801 #9s trajectory 0.005s
# t0 = 0.0
# tf = 0.005*(N-1)

problem = TrajOptProblem(sys,N,t0,tf,gradmode = True)

############################
#Add state bounds          #
############################




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
# problem.ubd = [np.array([-100]*12),np.array([100]*12)]		#self.extrapolation_factor = 2.5
# x0lower = vo.sub(single_traj_guess,[0]+[0.005]*14+[2]*15)
# x0upper = vo.add(single_traj_guess,[0]+[0.005]*14+[2]*15)
# problem.x0bd = [np.array(x0lower),np.array(x0upper)]
# problem.xfbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]


#####This is for setting 12
# diff = [2]*15
# diff[1] = 0.25 #small z change
# diff[2] = 0.8 #small torso angle change
# problem.xbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]
# problem.ubd = [np.array([-100]*12),np.array([100]*12)]
# #diff[2] = 0.1 #horizontal torso angle at the start of the traj
# problem.x0bd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]
# problem.xfbd = [np.array(vo.sub(single_traj_guess[0:15],diff) + [-2]*15),np.array(vo.add(single_traj_guess[0:15],diff) + [2]*15)]

######setting 14 & 15 & 16 & 17 & 18
# problem.xbd = [np.array([-0.2,0.4,-0.3] + [-math.pi]*12 + [-3]*15),np.array([5,1.1,0.3] + [math.pi]*12 + [3]*15)]
# problem.ubd = [np.array([-300]*12),np.array([300]*12)]
# problem.x0bd = [np.array([-0.05,0.4,-0.3] + [-math.pi]*12 + [-0.2]*15),np.array([0.05,1.1,0.3] + [math.pi]*12 + [0.2]*15)]
# problem.xfbd = [np.array([0.4,0.4,-0.3] + [-math.pi]*12 + [-2]*15),np.array([5,1.1,0.3] + [math.pi]*12 + [2]*15)]

##setting 19
problem.xbd = [np.array([-0.2,0.4,-0.3] + [-math.pi]*12 + [-3]*15),np.array([5,1.1,0.3] + [math.pi]*12 + [3]*15)]
problem.ubd = [np.array([-300]*12),np.array([300]*12)]
problem.x0bd = [np.array([-0.05,0.4,-0.3] + [-math.pi]*12 + [-0.2]*15),np.array([0.05,1.1,0.3] + [math.pi]*12 + [0.2]*15)]
problem.xfbd = [np.array([-0.2,0.4,-0.3] + [-math.pi]*12 + [-2]*15),np.array([5,1.1,0.3] + [math.pi]*12 + [2]*15)]


class anklePoseConstr(NonLinearPointConstr):
	def __init__(self):

		lb = np.array([-0.2,-1.0,-0.2,-1.0,-0.2,-1.0,-0.2,-1.0])
		ub = np.array([1.0]*8)
		self.robot = robosimian()
		NonLinearPointConstr.__init__(self,index = 0, nc = 8, nx = 30, nu = 12, np = 0 ,lb = lb, ub = ub, nG = 40)

	def __callg__(self,x, F, G, row, col, rec, needg):
		#first col of x is time
		self.robot.set_q_2D_(x[1:16])
		self.robot.set_q_dot_2D_(x[16:31])
		p = self.robot.get_ankle_positions()
		F[:] = np.array([p[0][1],p[0][2],p[1][1],p[1][2],p[2][1],p[2][2],p[3][1],p[3][2]])
		if needg:
			r = [0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 + [7]*5 
			c = [2,3,4,5,6] + [2,3,4,5,6] + [2,3,7,8,9] + [2,3,7,8,9] + [2,3,10,11,12] + [2,3,10,11,12] + [2,3,13,14,15] + [2,3,13,14,15]
			if rec:
				row[:] = r
				col[:] = c
			partial_Jp = self.robot.compute_Jp_Partial()
			Gs = []
			for (i,j) in zip(r,c):
				Gs.append(partial_Jp[i,j-1])
			G[:] = Gs

class cyclicConstr(LinearConstr):
	def __init__(self):
		global N
		dmin = 0.3
		dmax = 1.0
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
		#Remain same for the other state dimensions
		for i in range(nX):
			if i > 0:
				A[i,i] = 1.0
				A[i,i+(N-1)*(nX+nU+nP)] = -1.0
		LinearConstr.__init__(self,A,lb = lb,ub = ub)

#32 x (181*42)
class positiveTranslationConstr(LinearConstr):
	def __init__(self):
		global N
		dmin = 0.4
		dmax = 2000000.0
		nX = 30
		nU = 12
		nP = 0
		A = np.zeros((nX,N*(nX+nU+nP)))
		lb = np.ones(nX)*-1.0
		ub = np.ones(nX)
		#enough x translation
		A[0,0] = -1.0
		A[0,0+(N-1)*(nX+nU+nP)] = 1.0
		lb[0] = dmin
		ub[0] = dmax
		LinearConstr.__init__(self,A,lb = lb,ub = ub)

class enoughTranslationCost(NonLinearObj):
	def __init__(self):
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.d = 5.0
		self.C = 1.0#100.0
		global N
		self.N = N
		NonLinearObj.__init__(self,nsol = self.N*(self.nX+self.nU+self.nP),nG = 2  )

	def __callg__(self,x, F, G, row, col, rec, needg):
		x0 = x[0]
		xf = x[(self.N-1)*(self.nX+self.nU+self.nP)]
		if (xf-x0) >= self.d:
			F[:] = 0
		else:
			F[:] = self.C*(xf-x0-self.d)**2

		if needg:
			col_entries = [0,(self.N-1)*(self.nX+self.nU+self.nP)]
			if (xf-x0) >= self.d:
				Gs = [0.0,0.0]
			else:
				Gs = [-self.C*2.0*(xf-x0-self.d),self.C*2.0*(xf-x0-self.d)]
			G[:] = Gs
			if rec:
				row[:] = [0,0]
				col[:] = col_entries



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
		#setting 16 - 18
		#self.scale = 100.0
		#setting 17 run 3
		self.scale = 500.0#5000.0 # 500.0
		#TODO, need to modify the gradient for adding small_C
		self.small_C = 0.01
	def __callg__(self,x, F, G, row, col, rec, needg):
		effort_sum = 0.0
		#print('--------')
		for i in range(self.N):
			for j in range(self.NofJoints):
				effort_sum = effort_sum + (x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
					x[i*(self.nX+self.nU+self.nP)+self.first_u+j])**2
				# print(effort_sum)
		if math.fabs(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C) < 1e-10:
			F[:] = 0
		else:
			F[:] = effort_sum/math.fabs(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)/self.scale
		if needg:
			Gs = []
			nonzeros = [0]
			# for i in range(self.N):
			# 	for j in range(self.first_q_dot):
			# 		G[i*(self.nX+self.nU+self.nP)+j] = 0.0

			if x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C > 0:
				Gs.append(effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)**2)
			else:
				Gs.append(-effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)**2)
			d = math.fabs(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)
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
			if x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C > 0:
				Gs.append(-effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)**2)
			else:
				Gs.append(effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)**2)

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
			G[:] = vo.div(Gs,self.scale)
			if rec:
				row[:] = [0]*len(nonzeros)
				col[:] = nonzeros


#target velocity
Q = np.zeros(30)
Q[1] = 0.01
Q[2] = 0.01
Q[15] = 10.0
xbase = np.zeros(30)
xbase[2] = 0.0
xbase[1] = 0.9 #0.63# 0.9
xbase[15] = 0.3
targetVelocityCost = LqrObj(Q = Q, R = np.ones(12)*0.00001,xbase = xbase)

#zeroCost = LqrObj(Q= np.array([0.0]*15+[1.0]*15))
zeroCost = LqrObj(Q= np.array([0.0]*30))
#periodic cost
class periodicCost(NonLinearObj):
	def __init__(self):
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.C = 0.1
		global N
		self.N = N
		self.period = 60 #3s, 0.05dt, --- 60 interval
		self.state_length = self.nX+self.nU+self.nP
		NonLinearObj.__init__(self,nsol = self.N*(self.nX+self.nU+self.nP),nG = 12*self.N)

	def __callg__(self,x, F, G, row, col, rec, needg):
		cost = 0.0
		for i in range(self.N - self.period):
			cost += np.linalg.norm(x[i*(self.state_length)+3:i*(self.state_length)+15] - \
				x[(i+self.period)*(self.state_length)+3:(i+self.period)*(self.state_length)+15])**2
			#not adding velocity because initial velocity is more bounded..
			# cost += np.linalg.norm(x[i*(self.state_length)+18:i*(self.state_length)+30] - \
			# 	x[(i+self.period)*(self.state_length)+18:(i+self.period)*(self.state_length)+30])**2

		F[:] = self.C*cost

		if needg:
			col_entries = []
			Gs = []
			#add the first period entries
			for i in range(self.period):
				for j in range(12):
					col_entries.append(i*self.state_length+3+j)
					Gs.append((2.0*(x[i*self.state_length+3+j] - x[(i+self.period)*self.state_length+3+j]))*self.C)

			for i in range(self.N - self.period*2):
				for j in range(12):
					iterator = i + self.period
					col_entries.append(iterator*self.state_length+3+j)
					Gs.append((2.0*(x[iterator*self.state_length+3+j] - x[(iterator+self.period)*self.state_length+3+j]) - \
						2.0*(x[(iterator-self.period)*self.state_length+3+j] - x[iterator*self.state_length+3+j]))*self.C)
			for i in range(self.period):
				for j in range(12):
					iterator = i + (self.N - self.period)
					col_entries.append(iterator*self.state_length+3+j)
					Gs.append((-2.0*(x[(iterator-self.period)*self.state_length+3+j] - x[iterator*self.state_length+3+j]))*self.C)			
					
			G[:] = Gs
			if rec:
				row[:] = [0]*12*self.N
				col[:] = col_entries



##############################
#set the problem constraints #
##############################


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
# c = transportationCost()
# constr1 = cyclicConstr()
# problem.addNonLinearObj(c)
# problem.addLinearConstr(constr1)

#settting 14 & 15
# c = transportationCost()
# problem.addNonLinearObj(c)

# #setting 16,17, 18
# constr1 = anklePoseConstr()
# c = transportationCost()
# problem.addNonLinearObj(c) 
# problem.addNonLinearPointConstr(constr1,path = True)
# #setting 17, 16 run4
# constr2 = positiveTranslationConstr()
# problem.addLinearConstr(constr2)

#setting 19,20
# c1 = transportationCost()
# c2 = enoughTranslationCost()
# constr1 = anklePoseConstr()
# problem.addNonLinearObj(c1)
# problem.addNonLinearObj(c2)
# problem.addNonLinearPointConstr(constr1,path = True)

#setting 21,24
constr1 = anklePoseConstr()
periodicCost = periodicCost()
problem.add_lqr_obj(targetVelocityCost)
problem.addNonLinearObj(periodicCost)
problem.addNonLinearPointConstr(constr1,path = True)

#setting 23
# c2 = enoughTranslationCost()
# problem.addNonLinearObj(c2)
# problem.addNonLinearPointConstr(constr1,path = True)



############################
#preprocess                #
############################

startTime = time.time()

#setting before 19
# problem.preProcess()
#setting 19
problem.pre_process(snopt_mode = False) #dyn_tol
print('preProcess took:',time.time() - startTime)


###############################
##Optimizer parameter setting##
###############################

#print_level 1 for SNOPT is verbose enough
#print_level 0-12 for IPOPT 5 is appropriate, 6 more detailed
# cfg = OptConfig(backend='snopt', print_file='temp_files/tmp.out', print_level = 1, opt_tol = 1e-4, fea_tol = 1e-4, major_iter = 5,iter_limit = 10000000)
# cfg = OptConfig(backend='ipopt', print_file='temp_files/tmp.out', print_level = 5, opt_tol = 1e-4, fea_tol = 1e-4, major_iter = 2,deriv_check = True)
# slv = OptSolver(problem, cfg)
#setting 12
#slv.solver.setWorkspace(4000000,5000000)
#setting 14,15,18
#slv.solver.setWorkspace(7000000,8000000)
# slv.solver.setWorkspace(12000000,12000000)

##setting for using knitros, test 16,17
##1014 maxIter,1023 is featol_abs  1027 opttol 1016 is the output mode; 1033 is whether to use multistart
##1015 is the output level 1003 is the algorithm 
##1022 is the feastol_relative 1027 is opttol
##1006 is KN_PARAM_BAR_FEASIBLE 0 vs 3
##1097 KN_PARAM_INITPENALTY (default 1), 1080 derivative check, 1082 derivative check tolerence
##1049 adopt flexible penalty parameter in the merit function to weight feasibility vs optimality: 1:use a single parameter 2:flexible
#1105:knitro output name, 1047 path name
#1003: algorithm. #4 is sqp
#1004: how mu changes..
##debug information 1032
options = {'1105':'run8.log','1014':1000,'1023':1e-3,'1016':2,'1033':0,'1003':0,'1022':1e-3,'1027':1e-3,'1006':0,'1049':0,'1080':0,'1082':1e-4,'1004':4,\
	'history':True} #'1105':'run2.log','1047':'temp_files/',
cfg = OptConfig(backend = 'knitro', **options)
slv = OptSolver(problem, cfg)


###############################
##Initial Guess              ##
###############################

#These are the force and torque that could support the robot in place.
# single_traj_guess = [0,0.915,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6]+[0.0]*15
# single_u_guess = configs.u_augmented_mosek
###setting 1-13
# traj_guess = []
# u_guess = []
# for i in range(N):
# 	tmp = copy.copy(single_traj_guess)
# 	tmp[0] = i*0.002
# 	traj_guess.append(tmp)
# 	u_guess.append(single_u_guess)
#guess = problem.genGuessFromTraj(X= np.array(traj_guess), U=np.array(u_guess), t0 = 0, tf = tf)

###setting 14,15,16,18
#traj_guess = np.hstack((np.load('results/PID_trajectory/4/q_init_guess.npy'),np.load('results/PID_trajectory/4/q_dot_init_guess.npy')))
#u_guess = np.load('results/PID_trajectory/4/u_init_guess.npy')
# # traj_guess = np.load('results/16/run1/solution_x61.npy')
# # u_guess = np.load('results/16/run1/solution_u61.npy')
# guess = problem.genGuessFromTraj(X= traj_guess, U= u_guess, t0 = 0, tf = tf)

#setting 17,19,21
# traj_guess = np.load('results/PID_trajectory/3/x_init_guess.npy')
# u_guess = np.load('results/PID_trajectory/3/u_init_guess.npy')

# traj_guess = np.load('results/19/run2/solution_x11.npy')
# u_guess = np.load('results/19/run2/solution_u11.npy')

#setting 21, run2 ,24
# traj_guess = np.hstack((np.load('results/PID_trajectory/4/q_init_guess.npy'),np.load('results/PID_trajectory/4/q_dot_init_guess.npy')))
# u_guess = np.load('results/PID_trajectory/4/u_init_guess.npy')


#setting 22,

# traj_guess = np.hstack((np.load('results/PID_trajectory/7/q_history.npy')[200:2001],np.load('results/PID_trajectory/7/q_dot_history.npy')[200:2001]))
# u_guess = np.load('results/PID_trajectory/7/u_history.npy')[200:2001]

# traj_guess = np.hstack((np.load('results/PID_trajectory/4/q_history.npy')[200:2001],np.load('results/PID_trajectory/4/q_dot_history.npy')[200:2001]))
# u_guess = np.load('results/PID_trajectory/4/u_history.npy')[200:2001]


traj_guess = np.load('results/27/run2/solution_x401.npy')
u_guess = np.load('results/27/run2/solution_u401.npy')
guess = problem.genGuessFromTraj(X= traj_guess, U= u_guess, t0 = 0, tf = tf)




###############################
###save initial guess        ##
###############################
save_path = 'temp_files/'
save_flag = True
if save_flag:
	parsed_result = problem.parse_f(guess)
	for key, value in parsed_result.items() :
		print(key,value,np.shape(value))

	#np.save(save_path + 'knitro_obj0.npy',np.array([0.0]))
	np.save(save_path + 'knitro_obj0.npy',parsed_result['obj'])
	dyn_constr = np.array(parsed_result['dyn']).flatten()
	# ankle_constr = parsed_result['path'][0].flatten()
	# np.save('temp_files/knitro_con0.npy',np.concatenate((dyn_constr,ankle_constr,np.array([0.0]))))
	np.save(save_path + 'knitro_con0.npy',np.concatenate((dyn_constr,np.array([0.0]))))


###############################
###save solutions            ##
###############################
startTime = time.time()

### setting for using SNOPT
# iteration = 5
# rst = slv.solve_guess(guess)
# sol = problem.parse_sol(rst.sol.copy())
# np.save('temp_files/solution_u'+str(iteration),sol['u'])
# np.save('temp_files/solution_x'+str(iteration),sol['x'])
# print(str(iteration)+ 'iterations completed')
# for i in range(19):
# 	iteration += 5
# 	rst = slv.solver.solve_more(5)
# 	sol = problem.parse_sol(rst.sol.copy())
# 	np.save('temp_files/solution_u'+str(iteration),sol['u'])
# 	np.save('temp_files/solution_x'+str(iteration),sol['x'])
# 	print(str(iteration)+ 'iterations completed')


##setting for using Knitro
rst = slv.solve_guess(guess)
initial_i = 401
i = initial_i
for history in rst.history:
	if (i%10 == 0):
		sol = problem.parse_sol(history['x'])
		np.save(save_path + 'solution_u'+str(i+1)+'.npy',sol['u'])
		np.save(save_path + 'solution_x'+str(i+1)+'.npy',sol['x'])

		### This saves everything from the optimizers
		np.save(save_path + 'knitro_obj'+str(i+1)+'.npy',np.array(history['obj']))
		np.save(save_path + 'knitro_con'+str(i+1)+'.npy',history['con'])

		### 
		# result_0 = problem.genGuessFromTraj(X= sol['x'], U= sol['u'], t0 = 0, tf = tf)
		# parsed_result = problem.parse_f(result_0)
		# np.save('temp_files/solverlib_obj.npy',np.array(parsed_result['obj']))
		# np.save('temp_files/solverlib_con.npy',parsed_result['path'][0])
	
	i += 1

i = i - initial_i

sol = problem.parse_sol((rst.history[i])['x'])
np.save(save_path + 'solution_u'+str(i+initial_i)+'.npy',sol['u'])
np.save(save_path + 'solution_x'+str(i+initial_i)+'.npy',sol['x'])

### This saves everything from the optimizer
np.save(save_path + 'knitro_obj'+str(i+initial_i)+'.npy',np.array(history['obj']))
np.save(save_path + 'knitro_con'+str(i+initial_i)+'.npy',history['con'])

print('Took', time.time() - startTime)
print("========results=======")