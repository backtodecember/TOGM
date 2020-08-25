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
from scipy.interpolate import CubicSpline

#Parameters:
global N_of_control_pts
N_of_control_pts = 30
global N
N = 181 #9s trajectory 0.05
terrain = 1


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
		# a = self.robot.getDyn(x,u) #This is a numpy column 2D vector N*1
		# return np.concatenate([x[15:30],a]) #1D numpy array
		return np.zeros(30)

	def jac_dyn(self, t, x, u, p=None):
		# a,J_tmp = self.robot.getDynJac(x,u)
		# a = np.concatenate([x[15:30],np.ravel(a)])		
		# J = np.zeros((30,1+30+12))
		# J[:,1:43] = J_tmp
		# return a,J
		return np.zeros(30),np.zeros((30,43))

class EESpline(AddX):
	def __init__(self):
		global N_of_control_pts
		self.n = 2*4*N_of_control_pts 
		self.lb = np.array(([-2.0]*N_of_control_pts + [-0.3]*N_of_control_pts)*4)
		self.ub = np.array(([10.0]*N_of_control_pts + [1.0]*N_of_control_pts)*4)
		AddX.__init__(self, n = self.n, lb = self.lb, ub = self.ub)

class EESpline2(AddX):
	def __init__(self):
		global N_of_control_pts
		self.n = 4*N_of_control_pts 
		self.lb = np.array(([-1.0]*N_of_control_pts)*4)
		self.ub = np.array(([1.0]*N_of_control_pts)*4)
		AddX.__init__(self, n = self.n, lb = self.lb, ub = self.ub)


############################
#Initialize the problem    #
############################

sys = Robosimian()
t0 = 0.0
tf = 0.05*(N-1)

angle_spline_flag = True
if angle_spline_flag:
	EE = EESpline2()
	problem = TrajOptProblem(sys,N,t0,tf,gradmode = True, addx = EE)
else:
	EE = EESpline()
	problem = TrajOptProblem(sys,N,t0,tf,gradmode = True, addx = EE)

############################
#Add state bounds          #
############################
#TODO: Add other terrains
degree10 = math.pi*10.0/180.0
degree20 = math.pi*10.0/180.0
self.tana = math.tan(degree10)
self.cosa = math.cos(degree10)
if terrain == 0:
	problem.xbd = [np.array([-0.2,0.4,-0.3] + [-math.pi*1.5]*12 + [-3]*15),np.array([5,1.2,0.3] + [math.pi*1.5]*12 + [3]*15)]
	problem.ubd = [np.array([-300]*12),np.array([300]*12)]
	problem.x0bd = [np.array([-0.05,0.4,-0.3] + [-math.pi*1.5]*12 + [-2.0]*15),np.array([0.05,1.1,0.3] + [math.pi*1.5]*12 + [2.0]*15)]
	problem.xfbd = [np.array([-0.2,0.4,-0.3] + [-math.pi]*12 + [-2]*15),np.array([5,1.1,0.3] + [math.pi]*12 + [2]*15)]

elif terrain == 1:
	#This is assuming that the target speed of the torso is 0.4m/s on the slope
	problem.xbd = [np.array([-0.2,0.4,-0.3-degree10] + [-math.pi*1.5]*12 + [-3]*15),np.array([5,1.2 + math.sin(degree10)*3.6,0.3-degree10] + [math.pi*1.5]*12 + [3]*15)]
	problem.ubd = [np.array([-300]*12),np.array([300]*12)]
	problem.x0bd = [np.array([-0.05,0.4,-0.3-degree10] + [-math.pi*1.5]*12 + [-2.0]*15),np.array([0.05,1.1,0.3] + [math.pi*1.5]*12 + [2.0]*15)]
	problem.xfbd = [np.array([-0.2,0.4 + math.sin(degree10)*3.6,-0.3-degree10] + [-math.pi]*12 + [-2]*15),np.array([5,1.1 + math.sin(degree10)*3.6,0.3-degree10] + \
		[math.pi]*12 + [2]*15)]

#TODO: add terrain 5
class anklePoseConstr(NonLinearPointConstr):
	def __init__(self,terrain = 0):
		self.terrain = terrain
		lb = np.array([-0.3,-1.0,-0.3,-1.0,-0.3,-1.0,-0.3,-1.0])
		ub = np.array([1.0]*8)
		self.robot = robosimian()
		self.terrain > 0 and self.terrain < 5:
			if self.terrain == 1:
				self.tana = tan10
				self.cosa = cos10
		NonLinearPointConstr.__init__(self,index = 0, nc = 8, nx = 30, nu = 12, np = 0 ,lb = lb, ub = ub, nG = 40)

	def __callg__(self,x, F, G, row, col, rec, needg):
		#first col of x is time
		self.robot.set_q_2D_(x[1:16])
		self.robot.set_q_dot_2D_(x[16:31])
		p = self.robot.get_ankle_positions()

		if self.terrain == 0:
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
		elif self.terrain > 0 and self.terrain < 5:
			F[:] = np.array([(-self.tana*p[0][0] + p[0][1])*self.cosa,p[0][2]+degree10,\
				(-self.tana*p[1][0] + p[1][1])*self.cosa,p[1][2]+degree10,\
				(-self.tana*p[2][0] + p[2][1])*self.cosa,p[2][2]+degree10,\
				(-self.tana*p[3][0] + p[3][1])*self.cosa,p[3][2]+degree10])
			if needg:
				r = [0]*6 + [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6 + [6]*6 + [7]*6
				c = [1,2,3,4,5,6] + [1,2,3,4,5,6] + [1,2,3,7,8,9] + [1,2,3,7,8,9] + [1,2,3,10,11,12] + [1,2,3,10,11,12] + [1,2,3,13,14,15] + [1,2,3,13,14,15]
				if rec:
					row[:] = r
					col[:] = c
				partial_Jp = self.robot.compute_Jp_Partial3()
				Gs = []
				for (i,j) in zip(r,c):
					if i%2 == 0:
						limb = int(i/2)
						Gs.append(self.cosa*partial_Jp[limb*3+1,j-1] - self.tana*self.cosa*partial_Jp[limb*3+0,j-1])
					else:
						limb = int((i-1)/2)
						Gs.append(partial_Jp[limb*3+2,j-1])
				G[:] = Gs

class periodicCost(NonLinearObj):
	def __init__(self,C = 0.1, N = 60):
		global N_of_control_pts,angle_spline_flag
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.C = C
		global N
		self.N = N
		self.period = 60 #3s, 0.05dt, --- 60 interval
		self.state_length = self.nX+self.nU+self.nP
		if angle_spline_flag:
			self.nAddX =  4*N_of_control_pts
		else:
			self.nAddX =  2*4*N_of_control_pts
		NonLinearObj.__init__(self,nsol = self.N*(self.nX+self.nU+self.nP) + self.nAddX ,nG = 12*self.N)

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

def sp_predict(spx,spz,t):
	out = []
	xs = spx(t)
	zs = spz(t)
	for (x,z) in zip(xs,zs):
		out.append(x)
		out.append(z)
	return np.array(out)

def calculate_spline(control_pts,pred_indeces):
	"""
	Parameters:
	--------------
	control_pts: the 2D N control pts for the cubic spline. For N control pts, control_pts[0:N] are the x-values, control_pts[N:2N] are the z(y)-values
	indeces: where we would like to calculate the values for on the spline

	Return:
	------------- 
	pts: the predicted values on the spline, indicated by indeces
	J: the Jacobian w.r.t the control pts
	"""
	n = int(len(control_pts)/2)
	t = np.linspace(0,1,n)
	spx = CubicSpline(t,control_pts[0:n])
	spz = CubicSpline(t,control_pts[n:2*n])

	#indeces_scaled = indeces*sp.s[-1]
	out = sp_predict(spx,spz,pred_indeces)
	J = np.zeros((2*len(pred_indeces),2*n))
	eps = 1e-6			
	for k in range(2*n):
		add_vector = np.zeros(2*n)
		add_vector[k] = eps
		control_pts_tmp = control_pts + add_vector
		spx_tmp = CubicSpline(t,control_pts_tmp[0:n])
		spz_tmp = CubicSpline(t,control_pts_tmp[n:2*n])
		out_tmp = sp_predict(spx_tmp,spz_tmp,pred_indeces)
		J[:,k] = (out_tmp - out)/eps
	return out,J

def sp_predict2(spx,t):
	return spx(t)

def calculate_spline2(control_pts,pred_indeces):
	"""
	Parameters:
	--------------
	control_pts: 1 single spline control pts

	Return:
	------------- 
	pts: the predicted values on the spline, indicated by indeces
	J: the Jacobian w.r.t the control pts
	"""
	n = len(control_pts)
	t = np.linspace(0,1,n)
	sp = CubicSpline(t,control_pts)

	#indeces_scaled = indeces*sp.s[-1]
	out = sp_predict2(sp,pred_indeces)
	J = np.zeros((len(pred_indeces),2*n))
	eps = 1e-6			
	for k in range(n):
		add_vector = np.zeros(n)
		add_vector[k] = eps
		control_pts_tmp = control_pts + add_vector
		sp_tmp = CubicSpline(t,control_pts_tmp)
		out_tmp = sp_predict2(sp_tmp,pred_indeces)
		J[:,k] = (out_tmp - out)/eps
	return out,J

class EESplineConstraint(NonLinearConstr):
	def __init__(self):
		global N,N_of_control_pts
		self.gap = 1 #enforce constraint every 5 grid points
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.nAddX =  2*4*N_of_control_pts
		self.nc = int(8*((N-1)/self.gap+1))
		self.nc_4 = int(2*((N-1)/self.gap+1))
		self.state_length = N*(self.nX+self.nU+self.nP) 
		self.robot = robosimian()
		self.mesh_indeces = np.linspace(0,1,int((N-1)/self.gap+1))
		self.mesh_indeces_int = np.arange(0,N,self.gap)
		self.linear_indeces = np.arange(0,int((N-1)/self.gap+1),1)
		self.scale = 0.5

		NonLinearConstr.__init__(self, nsol = self.state_length + self.nAddX,nc = self.nc,lb = np.zeros(self.nc), ub = np.zeros(self.nc), \
			gradmode='user', nG =  95568 ) 
		#nG is obtained directly from looking at Gs #7696 (gap = 5) #37648 gap = 1 #95568 for gap =1 30 pts

	def __callg__(self,x, F, G, row, col, rec, needg):
		#note that in x, the first chunk elements are the states and controls, and the last chunk is the spline parameters
		state = x[0:self.state_length]
		AddX =  x[self.state_length:self.state_length+self.nAddX]
		#print(np.shape(x))
		constr = np.zeros(self.nc)
		Gs = []
		cols = []
		rows = []

		#calculate the splines
		#print(x[0:42])
		for i in range(4):
			sp_values,sp_derivatives = calculate_spline(AddX[i*int(self.nAddX/4):(i+1)*int(self.nAddX/4)],self.mesh_indeces)
			constr[i*self.nc_4:(i+1)*self.nc_4] = sp_values

			# print('control pts:',AddX[i*int(self.nAddX/4):(i+1)*int(self.nAddX/4)])
			# print(i)
			# print('sp_values:',sp_values)

			if needg:
				for iter_row in range(self.nc_4):
					for iter_col in range(int(self.nAddX/4)):
						Gs.append(sp_derivatives[iter_row,iter_col]*self.scale)
						rows.append(iter_row + i*self.nc_4)
						cols.append(iter_col + self.state_length + i*int(self.nAddX/4))

		fk_array = np.zeros(self.nc)
		for (i,j) in zip(self.mesh_indeces_int,self.linear_indeces):
			x_i = state[i*(self.nX+self.nU+self.nP):(i+1)*(self.nX+self.nU+self.nP)]
			self.robot.set_q_2D_(x_i[0:15])
			self.robot.set_q_dot_2D_(x_i[15:30])
			p = self.robot.get_ankle_positions()
			
			fk_array[2*j:2*j+2] = np.array([p[0][0],p[0][1]])
			fk_array[self.nc_4 + 2*j:self.nc_4 + 2*j+2] = np.array([p[1][0],p[1][1]])
			fk_array[self.nc_4*2 + 2*j:self.nc_4*2 + 2*j+2] = np.array([p[2][0],p[2][1]])
			fk_array[self.nc_4*3 + 2*j:self.nc_4*3 + 2*j+2] = np.array([p[3][0],p[3][1]])
			if needg:
				Jp = self.robot.compute_Jp_Partial2() #8 by 15
				Gs = Gs + (-1.0*Jp[0,[0,1,2,3,4,5]]*self.scale).tolist()
				Gs = Gs + (-1.0*Jp[1,[0,1,2,3,4,5]]*self.scale).tolist()
				rows = rows + [j*2]*6 + [j*2 + 1]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+3,\
					i*(self.nX+self.nU+self.nP)+4,i*(self.nX+self.nU+self.nP)+5]*2

				Gs = Gs + (-1.0*Jp[2,[0,1,2,6,7,8]]*self.scale).tolist()
				Gs = Gs + (-1.0*Jp[3,[0,1,2,6,7,8]]*self.scale).tolist()
				rows = rows + [j*2 + self.nc_4]*6 + [j*2 + 1 + self.nc_4]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+6,\
					i*(self.nX+self.nU+self.nP)+7,i*(self.nX+self.nU+self.nP)+8]*2

				Gs = Gs + (-1.0*Jp[4,[0,1,2,9,10,11]]*self.scale).tolist()
				Gs = Gs + (-1.0*Jp[5,[0,1,2,9,10,11]]*self.scale).tolist()
				rows = rows + [j*2 + self.nc_4*2]*6 + [j*2 + 1 + self.nc_4*2]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+9,\
					i*(self.nX+self.nU+self.nP)+10,i*(self.nX+self.nU+self.nP)+11]*2
				
				Gs = Gs + (-1.0*Jp[6,[0,1,2,12,13,14]]*self.scale).tolist()
				Gs = Gs + (-1.0*Jp[7,[0,1,2,12,13,14]]*self.scale).tolist()
				rows = rows + [j*2 + self.nc_4*3]*6 + [j*2 + 1 + self.nc_4*3]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+12,\
					i*(self.nX+self.nU+self.nP)+13,i*(self.nX+self.nU+self.nP)+14]*2

		#print('control pts:',x[self.state_length:-1])
		#print('constr',constr)
		constr = (constr - fk_array)*self.scale
		#print('fk_array:',fk_array)
		
		F[:] = constr
		if needg:
			G[:] = Gs
			if rec:
				row[:] = rows
				col[:] = cols
		return
		
class EESplineCost(NonLinearObj):
	def __init__(self):
		global N
		self.gap = 1 #enforce constraint every 5 grid points
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.nAddX =  2*4*10
		self.nc = int(8*((N-1)/self.gap+1))
		self.nc_4 = int(2*((N-1)/self.gap+1))
		self.state_length = N*(self.nX+self.nU+self.nP) 
		self.robot = robosimian()
		self.mesh_indeces = np.linspace(0,1,int((N-1)/self.gap+1))
		self.mesh_indeces_int = np.arange(0,N,self.gap)
		self.linear_indeces = np.arange(0,int((N-1)/self.gap+1),1)
		self.C = 0.002
		NonLinearObj.__init__(self,nsol = N*(self.nX+self.nU+self.nP) +self.nAddX,nG =N*(self.nX+self.nU+self.nP) +self.nAddX )

	def __callg__(self,x, F, G, row, col, rec, needg):
		#note that in x, the first chunk elements are the states and controls, and the last chunk is the spline parameters
		state = x[0:self.state_length]
		AddX =  x[self.state_length:self.state_length+self.nAddX]
		#print(np.shape(x))
		constr = np.zeros(self.nc)
		Gs = []
		cols = []
		rows = []

		for i in range(4):
			sp_values,sp_derivatives = calculate_spline(AddX[i*int(self.nAddX/4):(i+1)*int(self.nAddX/4)],self.mesh_indeces)
			constr[i*self.nc_4:(i+1)*self.nc_4] = sp_values
			if needg:
				for iter_row in range(self.nc_4):
					for iter_col in range(int(self.nAddX/4)):
						Gs.append(sp_derivatives[iter_row,iter_col])
						rows.append(iter_row + i*self.nc_4)
						cols.append(iter_col + self.state_length + i*int(self.nAddX/4))
		fk_array = np.zeros(self.nc)
		for (i,j) in zip(self.mesh_indeces_int,self.linear_indeces):
			x_i = state[i*(self.nX+self.nU+self.nP):(i+1)*(self.nX+self.nU+self.nP)]
			self.robot.set_q_2D_(x_i[0:15])
			self.robot.set_q_dot_2D_(x_i[15:30])
			p = self.robot.get_ankle_positions()
			
			fk_array[2*j:2*j+2] = np.array([p[0][0],p[0][1]])
			fk_array[self.nc_4 + 2*j:self.nc_4 + 2*j+2] = np.array([p[1][0],p[1][1]])
			fk_array[self.nc_4*2 + 2*j:self.nc_4*2 + 2*j+2] = np.array([p[2][0],p[2][1]])
			fk_array[self.nc_4*3 + 2*j:self.nc_4*3 + 2*j+2] = np.array([p[3][0],p[3][1]])
			if needg:
				Jp = self.robot.compute_Jp_Partial2() #8 by 15
				Gs = Gs + (-1.0*Jp[0,[0,1,2,3,4,5]]).tolist()
				Gs = Gs + (-1.0*Jp[1,[0,1,2,3,4,5]]).tolist()
				rows = rows + [j*2]*6 + [j*2 + 1]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+3,\
					i*(self.nX+self.nU+self.nP)+4,i*(self.nX+self.nU+self.nP)+5]*2

				Gs = Gs + (-1.0*Jp[2,[0,1,2,6,7,8]]).tolist()
				Gs = Gs + (-1.0*Jp[3,[0,1,2,6,7,8]]).tolist()
				rows = rows + [j*2 + self.nc_4]*6 + [j*2 + 1 + self.nc_4]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+6,\
					i*(self.nX+self.nU+self.nP)+7,i*(self.nX+self.nU+self.nP)+8]*2

				Gs = Gs + (-1.0*Jp[4,[0,1,2,9,10,11]]).tolist()
				Gs = Gs + (-1.0*Jp[5,[0,1,2,9,10,11]]).tolist()
				rows = rows + [j*2 + self.nc_4*2]*6 + [j*2 + 1 + self.nc_4*2]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+9,\
					i*(self.nX+self.nU+self.nP)+10,i*(self.nX+self.nU+self.nP)+11]*2
				
				Gs = Gs + (-1.0*Jp[6,[0,1,2,12,13,14]]).tolist()
				Gs = Gs + (-1.0*Jp[7,[0,1,2,12,13,14]]).tolist()
				rows = rows + [j*2 + self.nc_4*3]*6 + [j*2 + 1 + self.nc_4*3]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+12,\
					i*(self.nX+self.nU+self.nP)+13,i*(self.nX+self.nU+self.nP)+14]*2


		constr = constr - fk_array
		F[:] = 0.5*self.C*np.linalg.norm(constr)**2
		
		

		if needg:
			sJ = coo_matrix((Gs, (rows, cols)), shape=(self.nc,self.state_length+self.nAddX))
			G[:] = self.C*np.dot(sJ.todense().T,constr)
			if rec:
				row[:] = [0]*(self.state_length+self.nAddX)
				col[:] = np.arange(0,self.state_length+self.nAddX,1)

		return

class AngleSplineConstraint(NonLinearConstr):
	def __init__(self,scale = 0.5):
		global N,N_of_control_pts
		self.gap = 1 #enforce constraint every 5 grid points
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.nAddX =  4*N_of_control_pts
		self.nc = int(4*((N-1)/self.gap+1))
		self.nc_4 = int(((N-1)/self.gap+1))
		self.state_length = N*(self.nX+self.nU+self.nP) 
		self.robot = robosimian()
		self.mesh_indeces = np.linspace(0,1,int((N-1)/self.gap+1))
		self.mesh_indeces_int = np.arange(0,N,self.gap)
		self.linear_indeces = np.arange(0,int((N-1)/self.gap+1),1)
		self.scale = scale

		NonLinearConstr.__init__(self, nsol = self.state_length + self.nAddX,nc = self.nc,lb = np.zeros(self.nc), ub = np.zeros(self.nc), \
			gradmode='user', nG =  26064 ) #nG is obtained directly from looking at Gs #7696 gap = 5 #11584 gap = 1 #18824 gap = 1,controlpts = 20 #26064,30 control pts

	def __callg__(self,x, F, G, row, col, rec, needg):
		#note that in x, the first chunk elements are the states and controls, and the last chunk is the spline parameters
		state = x[0:self.state_length]
		AddX =  x[self.state_length:self.state_length+self.nAddX]
		#print(np.shape(x))
		constr = np.zeros(self.nc)
		Gs = []
		cols = []
		rows = []

		for i in range(4):
			sp_values,sp_derivatives = calculate_spline2(AddX[i*int(self.nAddX/4):(i+1)*int(self.nAddX/4)],self.mesh_indeces)
			constr[i*self.nc_4:(i+1)*self.nc_4] = sp_values

			if needg:
				for iter_row in range(self.nc_4):
					for iter_col in range(int(self.nAddX/4)):
						Gs.append(sp_derivatives[iter_row,iter_col]*self.scale)
						rows.append(iter_row + i*self.nc_4)
						cols.append(iter_col + self.state_length + i*int(self.nAddX/4))

		fk_array = np.zeros(self.nc)
		for (i,j) in zip(self.mesh_indeces_int,self.linear_indeces):
			x_i = state[i*(self.nX+self.nU+self.nP):(i+1)*(self.nX+self.nU+self.nP)]
			self.robot.set_q_2D_(x_i[0:15])
			self.robot.set_q_dot_2D_(x_i[15:30])
			p = self.robot.get_ankle_positions()
			
			fk_array[j] = p[0][2]
			fk_array[self.nc_4 + j] = p[1][2]
			fk_array[self.nc_4*2 + j] = p[2][2]
			fk_array[self.nc_4*3 + j] = p[3][2]

			if needg:
				Jp = self.robot.compute_Jp_Partial() #8 by 15
				Gs = Gs + (-1.0*Jp[1,[0,1,2,3,4,5]]*self.scale).tolist()
				rows = rows + [j]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+3,\
					i*(self.nX+self.nU+self.nP)+4,i*(self.nX+self.nU+self.nP)+5]

				Gs = Gs + (-1.0*Jp[3,[0,1,2,6,7,8]]*self.scale).tolist()
				rows = rows + [j + self.nc_4]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+6,\
					i*(self.nX+self.nU+self.nP)+7,i*(self.nX+self.nU+self.nP)+8]

				Gs = Gs + (-1.0*Jp[5,[0,1,2,9,10,11]]*self.scale).tolist()
				rows = rows + [j + self.nc_4*2]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+9,\
					i*(self.nX+self.nU+self.nP)+10,i*(self.nX+self.nU+self.nP)+11]
				
				Gs = Gs + (-1.0*Jp[7,[0,1,2,12,13,14]]*self.scale).tolist()
				rows = rows + [j + self.nc_4*3]*6
				cols = cols + [i*(self.nX+self.nU+self.nP),i*(self.nX+self.nU+self.nP)+1,i*(self.nX+self.nU+self.nP)+2,i*(self.nX+self.nU+self.nP)+12,\
					i*(self.nX+self.nU+self.nP)+13,i*(self.nX+self.nU+self.nP)+14]

		constr = (constr - fk_array)*self.scale
		
		F[:] = constr
		if needg:
			G[:] = Gs
			if rec:
				row[:] = rows
				col[:] = cols
		return

#TODO: implemented this, for "sand dome" terrain 
class terrainLQRCost(NonLinearObj):
	def __init__(self):
		pass

class terrainHeightCost(NonLinearPointObj):
	def __init__(self,a):
		self.height = 0.75
		self.weight = 0.01
		self.tan = math.tan(a)
		NonLinearPointObj.__init__(index = 0, nx = 30, nu = 12, np=0, gradmode='user', nG=2):
	def __callg__(self,x, F, G, row, col, rec, needg):
		F[:] = self.weight*(x[2] - x[1]*self.tan)**2
		if needg:
			if rec:
				row[:] = [0]*2
				col[:] = [1,2]
			G[:] = [2*self.weight*(x[2] - x[1]*self.tan)*(-self.tan), 2*self.weight*(x[2] - x[1]*self.tan)]

#TODO: add other terrains
if terrain == 0:
	#LQR, flat terain
	Q = np.zeros(30)
	Q[1] = 0.01 #0.01
	Q[2] = 0.01
	Q[15] = 100.0
	xbase = np.zeros(30)
	xbase[2] = 0.0
	xbase[1] = 0.75
	xbase[15] = 0.4
	R = np.ones(12)*0.00001  #0.00001

elif terrain == 1:
	Q = np.zeros(30)
	Q[2] = 0.01 #angle of the torso
	Q[15] = 100.0
	xbase = np.zeros(30)
	xbase[2] = -degree10
	xbase[15] = 0.4*cos10
	R = np.ones(12)*0.00001  #0.00001

targetVelocityCost = LqrObj(Q = Q,R=R,xbase = xbase)

##############################
#set the problem constraints #
##############################

#setting 21,24
constr1 = anklePoseConstr()
periodicCost = periodicCost(C = 0.1,N = 60)
problem.add_lqr_obj(targetVelocityCost)
problem.addNonLinearObj(periodicCost)
problem.addNonLinearPointConstr(constr1,path = True)

if angle_spline_flag:
	splineConstr = AngleSplineConstraint(scale = 0.5)
	problem.addNonLinearConstr(splineConstr)
else:
	splineConstr = EESplineConstraint()
	#splineCost = EESplineCost()
	# problem.addNonLinearObj(splineCost)
	problem.addNonLinearConstr(splineConstr)

if terrain == 1:
	terrainHeightCost = terrainHeightCost(degree10)
	problem.addNonLinearPointObj(terrainHeightCost,path = True)

############################
#preprocess                #
############################

startTime = time.time()
problem.pre_process(snopt_mode = False)
print('preProcess took:',time.time() - startTime)


###############################
##Deriv Check debug##
###############################
# cfg = OptConfig(backend='ipopt', print_file='temp_files2/tmp.out', print_level = 5, opt_tol = 1e-4, fea_tol = 1e-4, major_iter = 2,deriv_check = True)
# slv = OptSolver(problem, cfg)

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

run = '19'
options = {'1105':'run'+run+'.log','1014':1000,'1023':1e-3,'1016':2,'1003':0,'1022':1e-3,'1027':1e-3,'1006':0,'1049':0,'1080':0,'1082':1e-4,'1004':4,\
	'history':True}
cfg = OptConfig(backend = 'knitro', **options)
slv = OptSolver(problem, cfg)


###############################
##Initial Guess              ##
###############################

#setting 21, run2 ,24
path = 'results/PID_trajectory/13/'
traj_guess = np.hstack((np.load(path + 'q_init_guess.npy'),np.load(path + 'q_dot_init_guess.npy')))
u_guess = np.load(path + 'u_init_guess.npy')

# u_guess = []
# for i in range(N):
# 	u_guess.append([0]*12)
# u_guess = np.array(u_guess)

# for i in range(N):
# 	traj_guess[i,1] += 0.2


addX_guess = [np.array([0.0]*4*N_of_control_pts)]

# angle_list = []
# for i in range(N_of_control_pts):
# 	angle_list.append(0.2*math.sin(6.0*math.pi*i/N_of_control_pts))
# print(angle_list)
# addX_guess = [np.array(angle_list*4)]

# addX_guess = [np.array(np.linspace(0.235792,0.235792+ 2.7,N_of_control_pts).tolist() +  [-0.133401]*N_of_control_pts \
# 	+ np.linspace(-0.258711,-0.258711+2.7,N_of_control_pts).tolist() + [-0.129933]*N_of_control_pts + np.linspace(-0.771407,-0.771407+2.7,N_of_control_pts).tolist() + [-0.114612]*N_of_control_pts \
# 	+ np.linspace(0.748393,0.748393+2.7,N_of_control_pts).tolist() +[-0.101704]*N_of_control_pts)]


# below are for 10 control pts, and x,z control pts
# addX_guess = [np.array([0.235792]*10 +  [-0.133401]*10 + [-0.258711]*10 +  [-0.129933]*10 \
# 	+ [-0.771407]*10 + [-0.114612]*10 +  [0.748393]*10 +[-0.101704]*10)]

# addX_guess = [np.array(np.linspace(0.235792,0.235792+ 2.7,10).tolist() +  \
# 	[-0.133401] + [-0.133401+ 0.2*math.sin(math.pi/1.5*1.0),-0.133401,-0.133401]*3 \
# 	+ np.linspace(-0.258711,-0.258711+2.7,10).tolist() \
# 	+ [-0.129933] + [-0.129933,-0.129933+ 0.2*math.sin(math.pi/1.5*1.0),-0.129933]*3\
# 	+ np.linspace(-0.771407,-0.771407+2.7,10).tolist() \
# 	+ [-0.114612] + [-0.114612+ 0.2*math.sin(math.pi/1.5*1.0),-0.114612,-0.114612]*3\
# 	+ np.linspace(0.748393,0.748393+2.7,10).tolist() \
# 	+[-0.101704]+ [-0.101704,-0.101704 + 0.2*math.sin(math.pi/1.5*1.0),-0.101704]*3)]


# addX_guess = [np.array(np.linspace(0.235792,0.235792+ 2.7,10).tolist() +  \
# 	[-0.133401] + [-0.133401+ 0.3*math.sin(math.pi/1.5*1.0),-0.133401,-0.133401]*3 \
# 	+ np.linspace(-0.258711,-0.258711+2.7,10).tolist() \
# 	+ [-0.129933] + [-0.129933,-0.129933+ 0.3*math.sin(math.pi/1.5*1.0),-0.129933]*3\
# 	+ np.linspace(-0.771407,-0.771407+2.7,10).tolist() \
# 	+ [-0.114612] + [-0.114612+ 0.3*math.sin(math.pi/1.5*1.0),-0.114612,-0.114612]*3\
# 	+ np.linspace(0.748393,0.748393+2.7,10).tolist() \
# 	+[-0.101704]+ [-0.101704,-0.101704 + 0.3*math.sin(math.pi/1.5*1.0),-0.101704]*3)]


# traj_guess = np.load('results/32/run17/solution_x171.npy')
# u_guess = np.load('results/32/run17/solution_u171.npy')
# addX_guess = np.load('results/32/run17/solution_addx171.npy')

guess = problem.genGuessFromTraj(X= traj_guess[0:N,:], U= u_guess[0:N,:], addx = addX_guess, t0 = 0, tf = tf)

# guess = problem.genGuessFromTraj(X= traj_guess[0:N,:], U= u_guess[0:N,:],t0 = 0, tf = tf)



###############################
###save initial guess        ##
###############################
save_path = 'run'+run+'/'
save_flag = True
if save_flag:
	parsed_result = problem.parse_f(guess)
	# for key, value in parsed_result.items() :
	# 	print(key,value,np.shape(value))
	
	#np.save(save_path + 'knitro_obj0.npy',np.array([0.0]))
	np.save(save_path + 'knitro_obj0.npy',parsed_result['obj'])
	dyn_constr = np.array(parsed_result['dyn']).flatten()
	ankle_constr = parsed_result['path'][0].flatten()
	# EE_constr = parsed_result['nonlin'][0].flatten()
	# np.save(save_path+ 'knitro_con0.npy',np.concatenate((dyn_constr,ankle_constr,EE_constr)))

###############################
###save solutions            ##
###############################
startTime = time.time()


##setting for using Knitro
rst = slv.solve_guess(guess)
initial_i = 0
i = initial_i
for history in rst.history:
	if (i%10 == 0):
		sol = problem.parse_sol(history['x'])
		np.save(save_path + 'solution_u'+str(i+1)+'.npy',sol['u'])
		np.save(save_path + 'solution_x'+str(i+1)+'.npy',sol['x'])
		np.save(save_path + 'solution_addx'+str(i+1)+'.npy',sol['addx'])
		### This saves everything from the optimizers
		np.save(save_path + 'knitro_obj'+str(i+1)+'.npy',np.array(history['obj']))
		np.save(save_path + 'knitro_con'+str(i+1)+'.npy',history['con'])

		### 
		# result_0 = problem.genGuessFromTraj(X= sol['x'], U= sol['u'], t0 = 0, tf = tf)
		# parsed_result = problem.parse_f(result_0)
		# np.save('temp_files/solverlib_obj.npy',np.array(parsed_result['obj']))
		# np.save('temp_files/solverlib_con.npy',parsed_result['path'][0])
	
	i += 1

i = i - 1 - initial_i
sol = problem.parse_sol((rst.history[i])['x'])
np.save(save_path + 'solution_u'+str(i+initial_i + 1)+'.npy',sol['u'])
np.save(save_path + 'solution_x'+str(i+initial_i + 1)+'.npy',sol['x'])
np.save(save_path + 'solution_addx'+str(i+initial_i + 1)+'.npy',sol['addx'])
### This saves everything from the optimizer
np.save(save_path + 'knitro_obj'+str(i+initial_i + 1)+'.npy',np.array(history['obj']))
np.save(save_path + 'knitro_con'+str(i+initial_i + 1)+'.npy',history['con'])

print('Took', time.time() - startTime)
print("========results=======")