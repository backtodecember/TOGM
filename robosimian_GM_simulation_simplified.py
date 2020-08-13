"""
The file simulates a simplied 2D robosimian robot with only 4 active joints and 2 limbs.
"""
from robosimian_wrapper import robosimian
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from klampt.math import vectorops as vo
from klampt import vis
from copy import deepcopy,copy
import ctypes as ct
import configs
import pdb
import mosek
import multiprocessing as mp

#Zherong's package
from KlamptDiffNE import *
import pyDiffNE
import pickle


class robosimianSimulator:
	def __init__(self,q = np.zeros((7,1)), q_dot= np.zeros((7,1)) , dt = 0.01,dyn = 'own',print_level = 0, \
		augmented = True, RL = False, extrapolation = False, integrate_dt = 0.01):
		self.dyn = dyn
		self.q = q
		self.q_dot = q_dot
		self.print_level = print_level
		self.dt = dt
		self.integrate_dt = integrate_dt
		self.time = 0
		self.dof = 7
		self.integration = 'semi-Euler' #"semi-Euler"
		self.profile_computation = False

		if self.dyn == 'diffne':
			world = klampt.WorldModel()
			#TODO, change collision mesh size
			self.robot =  DiffNERobotModel(world,"Robosimian/robosimian_caesar_new_all_active.urdf",use2DBase=True)
			#specify fixed joints
			unusedJoints=["limb1_link0","limb1_link1","limb1_link2","limb1_link3","limb1_link5","limb1_link7",
			"limb2_link0","limb2_link1","limb2_link2","limb2_link3","limb2_link5","limb2_link7",
			"limb3_link0","limb3_link1","limb3_link2","limb3_link3","limb3_link4","limb3_link5","limb3_link6","limb3_link7",
			"limb4_link0","limb4_link1","limb4_link2","limb4_link3","limb4_link4","limb4_link5","limb4_link6","limb4_link7"]
			self.robot.eliminate_joints(unusedJoints)
			#now set the initial config
			self._set_DiffNE_q2(self.q)
			self._set_DiffNE_q_dot2(self.q_dot)

			#set up collision detection
			ees=[]
			for i in range(self.robot.body.nrJ()):
				if len(self.robot.body.children(i,True))==0:
					ee=DNE.EndEffectorBounds(i)
					zRange=DNE.Vec3d(0,0,0)
					if   self.robot.body.joint(i)._name.startswith("limb1"):
						zRange=DNE.Vec3d(-0.1,0,0)
					elif self.robot.body.joint(i)._name.startswith("limb2"):
						zRange=DNE.Vec3d( 0.1,0,0)
					elif self.robot.body.joint(i)._name.startswith("limb3"):
						zRange=DNE.Vec3d( 0.1,0,0)
					elif self.robot.body.joint(i)._name.startswith("limb4"):
						zRange=DNE.Vec3d(-0.1,0,0)
					ee.detectEndEffector(self.robot.body,zRange)
					ees.append(ee)


			#print('debug:',self.q,self,q_dot)
			#create the simulator, use RK1F mode, force-level MDP
			self.robot.create_simulator(accuracy=64,granular=True,mode = DNE.FORWARD_RK1F)     #this will use double
			#self.robot.create_simulator(accuracy=128,granular=True,mode = DNE.FORWARD_RK1F)    #this will use quadmath
			#self.robot.create_simulator(accuracy=512,granular=True,mode = DNE.FORWARD_RK1F)    #this will use MPFR

			#set the floor
			self.robot.set_floor([0,0,1,0],ees) # the list means Ax+By+Cz+D = 0. This basically sets a z = 0 plane.


		else:
			raise Exception('Wrong Dyn given')

		self.RL = RL
		if self.RL:
			self.ankle_poses = np.zeros((8,1))

	def getDynJac(self,x,control,continuous = False):
		"""
		Parameters:
		---------------
		x,u are 1d numpy arrays
		u is of dimension 4
		Return:
		---------------
		a: np 2D array
		DynJac: np array 14 x 18
		if continuous, also return the state (np 1D array)
		"""
		#flip axis for diffne
		if self.dyn == 'diffne':
			q = self._own_to_diffne(x[0:self.dof])
			q_dot = self._own_to_diffne(x[self.dof:2*self.dof])
			u = -deepcopy(control)
			qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		if self.dyn == 'diffne':
			tau = [0,0,0] + u.tolist() #diffNE has control on all dofs
			# print(tau,len(tau),self.robot.num_DOF())
			qdqNext,Dqdq,Dtau=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=True,Dtau=True)
			a = (np.array(qdqNext[self.dof:2*self.dof]) - qdq[self.dof:2*self.dof])/self.dt
			#print('flag2')
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
			#calcualte DynJac
			Dqdq = np.array(Dqdq)
			Dtau = np.array(Dtau) 
			#x means q, q_dot, tau
			#dq_dotdx = np.hstack((Dqdq[15:30,:],Dtau[15:30,3:15]))
			dq_dotdx = np.hstack((np.zeros((self.dof,self.dof)),np.eye(self.dof),np.zeros((self.dof,4))))
			dadq = Dqdq[self.dof:2*self.dof,0:self.dof]/self.dt
			dadq_dot = (Dqdq[self.dof:2*self.dof,self.dof:2*self.dof] - np.eye(self.dof))/self.dt 
			dadtau = Dtau[self.dof:2*self.dof,3:self.dof]/self.dt
			dadx = np.hstack((dadq,dadq_dot,dadtau))

			DynJac = np.vstack((dq_dotdx,dadx))
			#now flip the axes back
			DynJac = self._own_to_diffne(J = DynJac)

		if not continuous:
			return a,DynJac
		else:
			if self.dyn == 'diffne':
				self.q = self._own_to_diffne(q = np.array(qdqNext[0:self.dof]))[np.newaxis].T
				self.q_dot = self._own_to_diffne(q = np.array(qdqNext[self.dof:2*self.dof]))[np.newaxis].T
				return a,DynJac,np.concatenate((self.q.ravel(),self.q_dot.ravel()))

	def getDynJacNext(self,x,control):
		"""
		The dynamics jocabian here is not da/dx, but dxnext/dx

		"""
		if self.dyn == 'own':
			raise RuntimeError('own dynamics does not have this implemented yet')
		
		q = self._own_to_diffne(x[0:self.dof])
		q_dot = self._own_to_diffne(x[self.dof:2*self.dof])
		u = -deepcopy(control)
		qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		tau = [0,0,0] + u.tolist() #diffNE has control on all dofs
		qdqNext,Dqdq,Dtau=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=True,Dtau=True)
		
		#calcualte DynJac
		Dqdq = np.array(Dqdq)
		Dtau = np.array(Dtau)
		#x means q, q_dot, tau
		DynJac = np.hstack((Dqdq,Dtau[:,3:self.dof]))
		#now flip the axes back
		DynJac = self._own_to_diffne(J = DynJac)

		return DynJac

	def getDyn(self,x,control,continuous = False):
		"""
		Parameters:
		---------------
		x,u are 1d vectors
		u is of dimension 4
		Return:
		---------------
		a: np 2D array
		if continuous, also return the state (np 1D array)
		"""
		#flip axis for diffne
		if self.dyn == 'diffne':
			q = self._own_to_diffne(x[0:self.dof])
			q_dot = self._own_to_diffne(x[self.dof:2*self.dof])
			u = -deepcopy(control)
			qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		if self.dyn == 'diffne':
			
			tau = [0,0,0] + u.tolist() #diffNE has control on all dofs

			qdqNext,_,_=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=False,Dtau=False)
			a = (np.array(qdqNext[self.dof:2*self.dof]) - np.array(qdq)[self.dof:2*self.dof])/self.dt
			
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
		if not continuous:
			return a
		else:
			if self.dyn == 'diffne':
				self.q = self._own_to_diffne(q = np.array(qdqNext[0:self.dof]))[np.newaxis].T
				self.q_dot = self._own_to_diffne(q = np.array(qdqNext[self.dof:2*self.dof]))[np.newaxis].T
				return a,np.concatenate((self.q.ravel(),self.q_dot.ravel()))


	def getConfig(self):
		"""
		return:
		----------
		1d numpy array
		"""

		return copy(self.q.ravel())

	def getVel(self):
		"""
		return:
		----------
		1d numpy array
		"""

		return copy(self.q_dot.ravel())

	def simulateOnce(self,u,continuous_simulation = False, SA = False, fixed = False):#debug,counter):
		"""
		u: 1D np array
		"""	
		if self.dyn == 'diffne':
			q = self._own_to_diffne(q = self.q.ravel())
			q_dot = self._own_to_diffne(q = self.q_dot.ravel())
			u1 = -deepcopy(u)
			qdq = q.tolist() + q_dot.tolist()
			tau = [0,0,0] + u1.tolist() #diffNE has control on all dofs

			qdqNext,_,_=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=False,Dtau=False)
			a = (np.array(qdqNext[self.dof:2*self.dof]) - np.array(qdq)[self.dof:2*self.dof])/self.dt
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
			if not continuous_simulation:
				return 
			else:
				if self.integration == 'semi-Euler':
					self.q = self._own_to_diffne(q = np.array(qdqNext[0:self.dof]))[np.newaxis].T
					self.q_dot = self._own_to_diffne(q = np.array(qdqNext[self.dof:2*self.dof]))[np.newaxis].T
				elif self.integration == 'Euler':
					self.q = self.q + self.q_dot*self.dt
					self.q_dot = self.q_dot + a*self.dt
				return

			return
			

	def simulateTraj(self,u_traj):
		"""
		u is k * 12 numpy matrix
		"""
		x_traj = np.ravel(self.q)
		for u in u_traj:
			self.simulateOnce(u,True)
			x_traj = np.vstack((x_traj,np.ravel(self.q)))

		return x_traj
	
	def getWorld(self):
		return self.robot.get_world()

	def getRobot(self):
		return self.robot

	def reset(self,q,q_dot = np.array([0]*15)):
		"""
		Parameters:
		numpy 1D array
		"""
		self.q= q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		return 

	def _terrainFunction(self,x,a):
		"""right now this is a flat terrain"""
		"""a is the angle around the z axis, when a = 0, ankle should be """
		"""upright, and the angle for the wrench space is also 0"""
		"""The positive angle for the data collected for wrench space is """
		"""actually about the -z axis.."""
		"""(When the data was collected, the 2D plane is the x-z plane..)"""
		terrainHeight = 0
		#terrainNormal = [0,1] will use this for curved slope
		if x[1] <= terrainHeight:
			depth = x[1] - terrainHeight
			if depth < self.terrain.material_range[0]:
				return "error"
			if math.fabs(a) > self.terrain.material_range[1]:
				return "error"
			angle = -a
			#TODO: fix the angle to the correct quadrant if it is > 2pi
			return [depth,angle,0] #the last is the terrain angle.. used to rotate WS
		return 

	def _ele_square(self,q):
		q_2 = []
		for ele in q:
			q_2.append(ele**2)
		return np.array(q_2, ndmin =2).T

	def _unvectorize(self,v,NofRows,NofCols):
		Q = np.zeros((NofRows,NofCols))
		for i in range(NofCols):
			for j in range(NofRows):
				Q[j,i] = v[i*NofRows+j]

		return Q

	def _zero_3D_vector(self,x):
		for i in self.fixed_joint_indicies:
			x[i] = 0.0

	def _check_q_bounds(self,q):
		q_clamp = np.zeros((15,1))
		for i in range(15):
			if q[i,0] > configs.q_high[i]:
				q_clamp[i,0] = configs.q_high[i]
			elif  q[i,0] < configs.q_low[i]:
				q_clamp[i,0] = configs.q_low[i]
			else:
				q_clamp[i,0] = q[i,0]

		return q_clamp

	def _check_q_dot_bounds(self,q):
		q_clamp = np.zeros((15,1))
		for i in range(15):
			if q[i,0] > configs.q_dot_high[i]:
				q_clamp[i,0] = configs.q_dot_high[i]
			elif  q[i,0] < configs.q_dot_low[i]:
				q_clamp[i,0] = configs.q_dot_low[i]
			else:
				q_clamp[i,0] = q[i,0]

		return q_clamp	

	def _own_to_diffne(self,q = None,J = None):
		"""
		parameters:
		--------------	
		q: 1d numpy array
		J: 2d numpy array
		"""
		if q is not None:		
			q_new = deepcopy(q)
			indeces = [1] + np.arange(3,7,1).tolist()
			for i in indeces:
				q_new[i] = -q_new[i]
			return q_new
		elif J is not None:
			J_new = deepcopy(J)
			indeces_row =  [1] + np.arange(3,7,1).tolist() + [8] + np.arange(10,14,1).tolist()
			for i in indeces_row:
				#each column corresponds to each dof
				J_new[i,:] = -J_new[i,:]

			indeces_col =  [1] + np.arange(3,7,1).tolist() + [8] + np.arange(10,14,1).tolist() + np.arange(14,18,1).tolist()
			for i in indeces_col:
				#each column corresponds to each dof
				J_new[:,i] = -J_new[:,i]
			
			return J_new

		else:
			raise RuntimeError('_own_to_diffne():wrong input for')


	def _set_DiffNE_q(self,q):
		"""
		Parameters:
		-------------
		q is a 1d numpy array
		"""
		for i in range(self.dof):
			self.robot.qdq[i] = q[i]
		return
	def _set_DiffNE_q2(self,q):
		"""
		Parameters:
		-------------
		q is a 2d numpy array
		"""
		for i in range(self.dof):
			self.robot.qdq[i] = q[i,0]		
		return
	def _set_DiffNE_q_dot(self,q_dot):
		"""
		Parameters:
		-------------
		q_dot is a 1d numpy array
		"""
		for i in range(self.dof):
			self.robot.qdq[i+self.dof] = q_dot[i]
		return
	def _set_DiffNE_q_dot2(self,q_dot):
		"""
		Parameters:
		-------------
		q_dot is a 2d numpy array
		"""
		for i in range(self.dof):
			self.robot.qdq[i+self.dof] = q_dot[i,0]
		return

	def q_2D_to_3D_(self,q):
		"""
		Parameters:
		q is list/1D array

		Return:
		list
		"""
		self.joint_indices_3D = [0,2,4,8,10,12,16,18,20,24,26,28,32,34,36]
		self.joint_indices_2D = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
		q_3D = [0]*38
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_3D[i] = q[j]
		return q_3D

if __name__=="__main__":

	# #q_2D = np.array(configs.q_staggered_augmented)[np.newaxis] #four feet on the ground at the same time
	# q_2D = np.array(configs.q_test17)[np.newaxis].T
	# # q_2D = np.array(configs.q_symmetric)[np.newaxis] #symmetric limbs
	# q_dot_2D = np.array([0.0]*15)[np.newaxis].T
	# simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = 0.05, dyn = 'own',print_level = 0,augmented = True,\
	# 	extrapolation = True,integrate_dt = 0.05)
	
	
	# u = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	# 	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])


	#simulator.simulateOnce(configs.u1)
	#t = simulator.getStaticTorques(np.array(configs.q_staggered_limbs + [0]*15))
	#print(t)
	#print(configs.u)
	#np.savetxt('staticTorque1',t)
	#simulator.simulate(9, fixed = True,visualize = True)
	#simulator.debugSimulation()
	# #Q = np.array(configs.q_staggered_augmented + [0]*15)
	# #Q[3] = Q[3] + 0.5
	# #U = np.array(configs.u_augmented_mosek)
	# #a,j = simulator.getDynJac(Q, U)

	
	##### debug for dynjac
	#q_2D = np.array(configs.q_test17)
	##q_2D = np.array([0.0,0.5,0.0]+[0.0,0.0,-0.8]+[0.0,0.0,0.8]+[0.0,0.0,-0.8]+[0.0,0.0,0.8])
	q_2D = np.array([0,-0.7,0]+[math.pi*0.9/2,-math.pi/2,math.pi*1.1/2,-math.pi*0.9/2,math.pi/2,-math.pi*1.1/2,math.pi*0.9/2,
		-math.pi/2,math.pi*1.1/2,-math.pi*0.9/2,math.pi/2,-math.pi*1.1/2])
	q_2D = -q_2D
	q_dot_2D = np.array([0.0]*15)
	x = np.hstack((q_2D,q_dot_2D))
	q_2D = q_2D[np.newaxis].T
	q_dot_2D = q_dot_2D[np.newaxis].T



	# u = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	# 5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])
	u = np.array([0.0]*12)
	simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = 0.05, dyn = 'own',print_level = 0,augmented = True,\
		extrapolation = True,integrate_dt = 0.05)
	simulator.debugSimulation()
	a1,J1 = simulator.getDynJac(x,u)
	
	simulator2 = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = 0.05, dyn = 'diffne',print_level = 0,augmented = True,\
		extrapolation = True,integrate_dt = 0.05)

	start_time = time.time()
	a2,J2 = simulator2.getDynJac(x,u)
	print('time elapsed:',time.time() - start_time)
	print('Own:',a1)
	print('diffne:',a2)
	print('Own:',J1[15:30,0:3])
	print('diffne J:',J2[15:30,0:3])