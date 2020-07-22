#main file planning trajectories
#using trajoptlib
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import logging
from trajoptlib.io import get_onoff_args
from trajoptlib import System, NonLinearPointObj, LqrObj,LinearObj,AddX
from trajoptlib import LinearConstr,NonLinearObj,NonLinearPointConstr,NonLinearConstr
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
from pycubicspline import Spline2D

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

class EESpline(AddX):
	def __init__(self):
		self.n = 2*4*10 #10 control points
		self.lb = np.array([-1,-0.2]*40)
		self.ub = np.array([10,1.0]*40)
		AddX.__init__(self, n = self.n, lb = self.lb, ub = self.ub)

############################
#Initialize the problem    #
############################
sys = Robosimian()
global N
N = 181 #9s trajectory 0.05s
t0 = 0.0
tf = 0.05*(N-1)
EE = EESpline()
problem = TrajOptProblem(sys,N,t0,tf,gradmode = True, AddX = EE)

############################
#Add state bounds          #
############################

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

#target velocity
Q = np.zeros(30)
Q[1] = 0.01
Q[2] = 0.01
Q[15] = 10.0
xbase = np.zeros(30)
xbase[2] = 0.0
xbase[1] = 0.9
xbase[15] = 0.3
targetVelocityCost = LqrObj(Q = Q, R = np.ones(12)*0.00001,xbase = xbase)

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

def calculate_spline()
	return

class EESplineConstraint(NonLinearConstr):
	def __init__(self):
		global N
		self.gap = 5 #enforce constraint every 5 grid points
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.nAddX =  2*4*10
		self.nc = 8*((N-1)/self.gap+1)
		self.state_length = N*(self.nX+self.nU+self.nP) 

		#the points at which the dynamics are enforced
		self.mesh_indeces = np.linspace(0,1,(N-1)/self.gap+1)

		NonLinearConstr.__init__(self, nsol = self.state_length + self.nAddx,nc = self.nc,lb = np.zeros(self.nc), ub = np.zeros(self.nc), \
			gradmode='user', nG = )

	def __init__(self,x, F, G, row, col, rec, needg):
		#note that in x, the first chunk elements are the states and controls, and the last chunk is the spline parameters
		state = x[0:self.state_length]
		AddX =  x[self.state_length:self.state_length+self.nAddX]
		constr = np.zeros(self.nc)
		#calculate the splines for 
		for i in range(4):
			sp_values = calculate_spline(AddX[i*(self.nAddX/4):(i+1)*(self.nAddX/4)],self.mesh_indeces)
			constr[i*self.nc/4:(i+1)*self.nc/4] = sp_values




##############################
#set the problem constraints #
##############################

#setting 21,24
constr1 = anklePoseConstr()
periodicCost = periodicCost()
problem.add_lqr_obj(targetVelocityCost)
problem.addNonLinearObj(periodicCost)
problem.addNonLinearPointConstr(constr1,path = True)

############################
#preprocess                #
############################

startTime = time.time()

#setting before 19
# problem.preProcess()
#setting 19
problem.pre_process(snopt_mode = False)
print('preProcess took:',time.time() - startTime)


###############################
##Optimizer parameter setting##
###############################

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
options = {'1105':'run1.log','1014':1000,'1023':1e-3,'1016':2,'1033':0,'1003':0,'1022':1e-3,'1027':1e-3,'1006':0,'1049':0,'1080':0,'1082':1e-4,'1004':6,\
	'history':True} #'1105':'run2.log','1047':'temp_files/',
cfg = OptConfig(backend = 'knitro', **options)
slv = OptSolver(problem, cfg)


###############################
##Initial Guess              ##
###############################

#setting 21, run2 ,24
traj_guess = np.hstack((np.load('results/PID_trajectory/4/q_init_guess.npy'),np.load('results/PID_trajectory/4/q_dot_init_guess.npy')))
u_guess = np.load('results/PID_trajectory/4/u_init_guess.npy')

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


##setting for using Knitro
rst = slv.solve_guess(guess)
i = 0
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

i = i - 1

sol = problem.parse_sol((rst.history[i])['x'])
np.save(save_path + 'solution_u'+str(i+1)+'.npy',sol['u'])
np.save(save_path + 'solution_x'+str(i+1)+'.npy',sol['x'])

### This saves everything from the optimizer
np.save(save_path + 'knitro_obj'+str(i+1)+'.npy',np.array(history['obj']))
np.save(save_path + 'knitro_con'+str(i+1)+'.npy',history['con'])

print('Took', time.time() - startTime)
print("========results=======")