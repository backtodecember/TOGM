#The 2D simulator traversing flat granular media terrain
#Use x-z 2D plane
#This file includes both MPQP and CVXPY as solvers
#only use the V formulation
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
	def __init__(self,q = np.zeros((18,1)), q_dot= np.zeros((18,1)) , dt = 0.01,dyn = 'own',print_level = 0, \
		augmented = True, RL = False, extrapolation = False, integrate_dt = 0.01):
		self.dyn = dyn
		self.q = q
		self.q_dot = q_dot
		self.print_level = print_level
		self.dt = dt
		self.integrate_dt = integrate_dt
		self.time = 0
		self.dof = 15
		self.integration = 'semi-Euler' #"semi-Euler"
		self.profile_computation = False
		self.D = 18 

		if self.dyn == 'diffne':
			world = klampt.WorldModel()
			#TODO, change collision mesh size
			self.robot =  DiffNERobotModel(world,"Robosimian/robosimian_caesar_new_all_active.urdf",use2DBase=False)
			#specify fixed joints
			unusedJoints=["limb1_link0","limb1_link1","limb1_link3","limb1_link5","limb1_link7",
			"limb2_link0","limb2_link1","limb2_link3","limb2_link5","limb2_link7",
			"limb3_link0","limb3_link1","limb3_link3","limb3_link5","limb3_link7",
			"limb4_link0","limb4_link1","limb4_link3","limb4_link5","limb4_link7"]
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
		u is of dimension 12
		Return:
		---------------
		a: np 2D array
		DynJac: np array 30 x 42
		if continuous, also return the state (np 1D array)
		"""
		#flip axis for diffne
		if self.dyn == 'diffne':
			q = self._own_to_diffne(x[0:self.D])
			q_dot = self._own_to_diffne(x[self.D:2*self.D])
			u = -deepcopy(control)
			qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		if self.dyn == 'diffne':
			tau = [0,0,0,0,0,0] + u.tolist() #diffNE has control on all dofs

			#print('flag1')
			# print(tau,len(tau),self.robot.num_DOF())
			qdqNext,Dqdq,Dtau=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=True,Dtau=True)
			a = (np.array(qdqNext[self.D:2*self.D]) - qdq[self.D:2*self.D])/self.dt
			#print('flag2')
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
			#calcualte DynJac
			Dqdq = np.array(Dqdq) #30-by-30
			Dtau = np.array(Dtau) #30-by-15
			#x means q, q_dot, tau
			#dq_dotdx = np.hstack((Dqdq[15:30,:],Dtau[15:30,3:15]))
			dq_dotdx = np.hstack((np.zeros((self.D,self.D)),np.eye(self.D),np.zeros((self.D,12))))

			dadq = Dqdq[self.D:2*self.D,0:self.D]/self.dt
			dadq_dot = (Dqdq[self.D:2*self.D,self.D:2*self.D] - np.eye(self.D))/self.dt 
			dadtau = Dtau[self.D:2*self.D,6:self.D]/self.dt
			dadx = np.hstack((dadq,dadq_dot,dadtau))

			DynJac = np.vstack((dq_dotdx,dadx))
			#now flip the axes back
			DynJac = self._own_to_diffne(J = DynJac)

		if not continuous:
			return a,DynJac
		else:
			if self.dyn == 'diffne':
				self.q = self._own_to_diffne(q = np.array(qdqNext[0:self.D]))[np.newaxis].T
				self.q_dot = self._own_to_diffne(q = np.array(qdqNext[self.D:2*self.D]))[np.newaxis].T
				return a,DynJac,np.concatenate((self.q.ravel(),self.q_dot.ravel()))

	def getDynJacNext(self,x,control):
		"""
		The dynamics jocabian here is not da/dx, but dxnext/dx

		"""
		if self.dyn == 'own':
			raise RuntimeError('own dynamics does not have this implemented yet')
		
		q = self._own_to_diffne(x[0:self.D])
		q_dot = self._own_to_diffne(x[self.D:2*self.D])
		u = -deepcopy(control)
		qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		tau = [0,0,0,0,0,0] + u.tolist() #diffNE has control on all dofs
		qdqNext,Dqdq,Dtau=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=True,Dtau=True)
		
		#calcualte DynJac
		Dqdq = np.array(Dqdq) #30-by-30
		Dtau = np.array(Dtau) #30-by-15
		#x means q, q_dot, tau
		DynJac = np.hstack((Dqdq,Dtau[:,6:self.D]))
		#now flip the axes back
		DynJac = self._own_to_diffne(J = DynJac)

		return DynJac

	def getDyn(self,x,control,continuous = False):
		"""
		Parameters:
		---------------
		x,u are 1d vectors
		u is of dimension 12
		Return:
		---------------
		a: np 2D array
		if continuous, also return the state (np 1D array)
		"""
		#flip axis for diffne
		if self.dyn == 'diffne':
			q = self._own_to_diffne(x[0:self.D])
			q_dot = self._own_to_diffne(x[self.D:2*self.D])
			u = -deepcopy(control)
			qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		if self.dyn == 'diffne':
			
			tau = [0,0,0,0,0,0] + u.tolist() #diffNE has control on all dofs

			qdqNext,_,_=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=False,Dtau=False)
			a = (np.array(qdqNext[self.D:2*self.D]) - np.array(qdq)[self.D:2*self.D])/self.dt
			
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
		if not continuous:
			return a
		else:
			if self.dyn == 'oiffne':
				self.q = self._own_to_diffne(q = np.array(qdqNext[0:self.D]))[np.newaxis].T
				self.q_dot = self._own_to_diffne(q = np.array(qdqNext[self.D:2*self.D]))[np.newaxis].T
				return a,np.concatenate((self.q.ravel(),self.q_dot.ravel()))

	# def getStaticTorques(self,x):
	# 	"""
	# 	Only available for own dynamics
	# 	"""
	# 	q = x[0:15]
	# 	q_dot = x[15:30]
	# 	self.q = q[np.newaxis].T
	# 	self.q_dot = q_dot[np.newaxis].T
	# 	self.robot.set_q_2D_(q)
	# 	self.robot.set_q_dot_2D_(q_dot)
	# 	u = 0
	# 	_,t = self.simulateOnce(u, fixed = True)
	# 	return t

	# def simulateOnceRL(self,u):
	# 	_ = self.simulateOnce(u,True,False,False)

	# 	return np.vstack((self.q,self.q_dot,self.ankle_poses))

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
			tau = [0,0,0,0,0,0] + u1.tolist() #diffNE has control on all dofs

			qdqNext,_,_=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=False,Dtau=False)
			a = (np.array(qdqNext[self.D:2*self.D]) - np.array(qdq)[self.D:2*self.D])/self.dt
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
			if not continuous_simulation:
				return 
			else:
				if self.integration == 'semi-Euler':
					self.q = self._own_to_diffne(q = np.array(qdqNext[0:self.D]))[np.newaxis].T
					self.q_dot = self._own_to_diffne(q = np.array(qdqNext[self.D:2*self.D]))[np.newaxis].T
				elif self.integration == 'Euler':
					self.q = self.q + self.q_dot*self.dt
					self.q_dot = self.q_dot + a*self.dt
				return

			return
			
	def simulate(self,total_time,plot = False, fixed = True,visualize = False):
		world = self.robot.get_world()
		if visualize:
			vis.add("world",world)
			vis.show()

		time.sleep(3)

		simulation_time = 0

		u1 = [6.08309021,0.81523653, 2.53641154, 5.83534863, 0.72158568, 2.59685143,\
				5.50487329, 0.54710471, 2.57836468 ,5.75260704 ,0.64075017, 2.51792186]
		u2 = [-33.85296297, -20.91288138, -0.9837711 , -33.41402916, -20.41161062 ,\
			-0.4201634 , -33.38633294 ,-20.41076484 , -0.44616818 ,-33.82823062,\
			-20.90921505 , -1.00117092]

		x_list = []
		u_list = []
		time_list = []
		#while passed_time < total_time:
		while simulation_time < total_time+0.0001:
			if visualize:
				vis.lock()
			# start = time.time()
			#self.simulateOnce(np.array(u[iteration_counter]))#,iteration_counter)
			#self.simulateOnce(u[iteration_counter%3])

			_, t = self.simulateOnce(u1,continuous_simulation = True, fixed = fixed)
			u_list.append(t)
			x_list.append(self.q.ravel().tolist()+self.q_dot.ravel().tolist())
			time_list.append(simulation_time)
			simulation_time = simulation_time + self.integrate_dt
			print(simulation_time)
			print('current q:',self.q)

			#time.sleep(self.dt*10)
			#time.sleep(0.001)

			if visualize:
				vis.unlock()
			#time.sleep(self.dt*1.0)
		if visualize:
			vis.kill()


		np.save('x_init_guess.npy',np.array(x_list))
		np.save('u_init_guess.npy',np.array(u_list))
		np.save('time_init_guess.npy',np.array(time_list))

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

	def debugSimulation(self):
		world = self.robot.get_world()
		vis.add("world",world)
		vis.show()

		vis.show()
		while vis.shown():
			vis.lock()
			#...do stuff to world... #this code is executed at approximately 10 Hz due to the sleep call
			vis.unlock()
			time.sleep(0.1)
		vis.kill()

	def reset(self,q,q_dot = np.array([0]*15)):
		"""
		Parameters:
		numpy 1D array
		"""
		self.q= q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		return 

	# def generateContacts(self):
	# 	##How to generate contact on a curved surface needs a bit more thoughts/care
	# 	"""Right not do not handle the case where a rigid body's angle is > pi/2.."""

	# 	#print(self.robot.get_ankle_positions())

	# 	contacts = []
	# 	limb_indices = []
	# 	NofContacts = 0

	# 	#loop through the 4 ankles
	# 	positions = self.robot.get_ankle_positions()

	# 	for i in range(4):
	# 		p = positions[i]
	# 		if self.RL:
	# 			self.ankle_poses[i*2,0] = p[1]
	# 			self.ankle_poses[i*2+1,0] = p[2]
	# 		# flag = False 
	# 		# if p[1] < self.terrain.material_range[0]:
	# 		# 	#p[1] = self.terrain.material_range[0]
	# 		# 	flag= True
	# 		# if p[2] > self.terrain.material_range[1]:
	# 		# 	#p[2] = self.terrain.material_range[1]
	# 		# 	flag = True
	# 		# if p[2] <  -self.terrain.material_range[1]:
	# 		# 	#p[2] = -self.terrain.material_range[1]
	# 		# 	flag = True
	# 		# if flag:
	# 		# 	print('One of more ankles are penetrating the terrain outside of database range,using extrapolation')
	# 		#if p[1] <= 0:
	# 		#even of not contact, still give a contact force 
	# 		if p[2] >= 0:
	# 			contact = [p[1],p[2],1,i,0] #the last element doesn't really mean anything, it's from the matlab program...
	# 		else:
	# 			if not self.augmented:
	# 				contact = [p[1],-p[2],-1,i,0]
	# 			else:
	# 				contact = [p[1],p[2],1,i,0]
	# 		if self.augmented:
	# 			contacts.append(contact)
	# 			limb_indices.append(i)
	# 			NofContacts += 1
	# 		else:
	# 			if p[1] <= 0:
	# 				contacts.append(contact)
	# 				limb_indices.append(i)
	# 				NofContacts += 1
	# 	#print(contacts)
	# 	return contacts,NofContacts,limb_indices

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
		#indeces = [1] + np.arange(6,18,1).tolist() #[6,7,8,9,10,11,12,13,14,15,16,17]
		if q is not None:		
			q_new = deepcopy(q)
			indeces = [1] + np.arange(6,18,1).tolist() #[3,4,5,6,7,8,9,10,11,12,13,14]
			for i in indeces:
				q_new[i] = -q_new[i]
			return q_new
		elif J is not None:
			J_new = deepcopy(J)
			indeces_row =  [1] + np.arange(6,18,1).tolist() + [19] + np.arange(24,36,1).tolist()
			for i in indeces_row:
				#each column corresponds to each dof
				J_new[i,:] = -J_new[i,:]

			indeces_col =  [1] + np.arange(6,18,1).tolist() + [19] + np.arange(24,36,1).tolist() + np.arange(36,48,1).tolist()
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