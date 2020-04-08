#The 2D simulator traversing flat granular media terrain
#Use x-z 2D plane
#This file includes both MPQP and CVXPY as solvers
#only use the V formulation
from robosimian_utilities import granularMedia
from robosimian_wrapper import robosimian
from MPQP import MPQP
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cvxpy as cp
from klampt.math import vectorops as vo
from klampt import vis
import multiprocessing as mp
from copy import deepcopy
from klampt.math import vectorops as vo
import ctypes as ct
class robosimianSimulator:
	def __init__(self,q = np.zeros((15,1)), q_dot= np.zeros((15,1)) , dt = 0.01, solver = 'cvxpy'):

		self.robot = robosimian()
		self.q = q
		self.q_dot = q_dot
		self.terrain = granularMedia(material = "sand")
		self.dt = dt
		self.time = 0

		self.Dx = 30 
		self.Du = 12
		self.Dwc = 12 
		self.Dlambda = 4*26
		self.solver = solver

		self.robot.set_q_2D_(self.q)
		self.robot.set_q_dot_2D(self.q_dot)
		#set up M1, which is used for scaling inertia w.r.t. granular media inertia during optimization
		#self.compute_pool = mp.Pool(4)
		ankle_density = 800
		ankle_radius = 0.0267
		ankle_length = 0.15
		ankle_mass = math.pi*ankle_radius**2*ankle_length*ankle_density
		ankle_rot_inertia = ankle_mass*ankle_radius**2/2.0
		self.ankle_inertia = np.array([[ankle_mass,0,0],[0,ankle_mass,0],[0,0,ankle_rot_inertia]])
		
		## TODO: there might exist a better way to compute M_2
		self.M_2 = np.zeros((4*3,4*3))#This is the mass of granular material...
		for i in range(4):
			self.M_2[0+i*3:3+i*3,0+i*3:3+i*3] = self.ankle_inertia ##same as the ankle mass....

		if self.solver == 'cvxpy':
			self.C = cp.Parameter((15,1)) #joint constraints
			self.D = cp.Parameter((15,12)) #contact jacobian
			self.M = cp.Parameter((15,15),PSD=True)
			self.Jp = cp.Parameter((12,15))
			#for each contact there are 26 lambdas
			self.x = cp.Variable((4*3+4*26,1)) #contact wrench and lambdas
			self.A = cp.Parameter((3*4,26*4))
			self.A2 = cp.Parameter((4,26*4))
			self.b2 = cp.Parameter((4,1))
			#self.obj = cp.Minimize(cp.quad_form(self.C+self.D@self.x[0:12],self.M)+\
			#	cp.quad_form(self.Jp*(self.C+self.D@self.x[0:12]),self.M_2))
			self.obj = cp.Minimize(cp.quad_form(self.C+np.dot(self.D,self.x[0:12]),self.M))

			self.constraints = [self.x[0:12] - np.dot(self.A,self.x[12:12+26*4]) == np.zeros((12,1)),\
				np.dot(self.A2,self.x[12:12+26*4]) <= self.b2,\
				-self.x[12:12+26*4] <= np.zeros((26*4,1))]
			self.prob = cp.Problem(self.obj, self.constraints)			
			#self.expr = cp.transforms.indicator(self.constraints)

		elif self.solver == 'mpqp':
			self.mpqp=MPQP("software")
			
		else:
			print("Wrong formulation specification")
			exit()
	
	def getDynJac(self,x,u):
		q_2D = x[0:15]
		q_dot_2D = x[15:30]
		self.q_3D = self.robot.set_q_2D_(q_2D)
		self.q_dot_3D = self.robot.set_q_dot_2D_(q_dot_2D)
		a,DynJac = self.simulateOnce(u)
		return self._3D_to_2D(a),DynJac

	def simulateOnce(self,u,continuous_simulation = False):#debug,counter):
		contacts,NofContacts,limb_indices = self.generateContacts()
		print(contacts)
		A = np.zeros((self.Dwc,self.Dlambda))
		A2 = np.zeros((4,self.Dlambda)) 
		b2 = np.zeros((4,1))
		Jp = self.robot.compute_Jp(contact_list=limb_indices)
		
		#gradient version..
		C, D = self.robot.compute_CD(u) # different from above, C = C*dt + Vk, D = D*dt

		######compute the wrench spaces serially #####

		for contact in contacts:
			add_A,Q4s = self.terrain.feasibleWrenchSpace(contact,self.robot.ankle_length,True)
			A[contact[3]*3:(contact[3]+1)*3,contact[3]*26:(contact[3]+1)*26] = add_A
			A2[contact[3],contact[3]*26:(contact[3]+1)*26] = np.ones((1,26))
			b2[contact[3]] = 1

		if NofContacts > 0:

			if self.solver == 'cvxpy':
				self.C.value = np.add(np.multiply(C,self.dt),self.q_dot) 
				self.D.value = np.multiply(D,self.dt)
				self.M.value = self.robot.get_mass_matrix()
				self.Jp.value = Jp
				self.A.value = A
				self.A2.value = A2
				self.b2.value = b2

				start_time = time.time()

				self.prob.solve(solver=cp.OSQP,verbose = False,warm_start = False,eps_abs = 1e-11,eps_rel = 1e-11,max_iter = 100000000)
				#self.prob.solve(solver=cp.OSQP,verbose = False,warm_start = False)#,eps_abs = 1e-12,eps_rel = 1e-12,max_iter = 10000000)
				#self.prob.solve(verbose = False,warm_start = True)
				#print(self.constraints[2].dual_value)
				#print('time:',time.time() - start_time)
				x_k = self.x.value[0:12]
				print('x_k',x_k)
				print("status:", self.prob.status)
			
				if continuous_simulation:
					self.q_dot = np.add(self.C.value,np.dot(self.D.value,x_k))

			elif self.solver == 'mpqp':

				# H =  np.zeros((12,12),dtype = np.float32)
				# g = np.zeros((1,12),dtype = np.float32)

				M = self.robot.get_mass_matrix()
				H = np.dot(D.T,np.dot(M,D))
				g = np.dot((self.q_dot.T + C.T),np.dot(M,D))
				g = g.T
				gw = np.dot(A.T,g)
				Hw = np.dot(A.T,np.dot(H,A))
				Hw.astype(np.float32)
				gw.astype(np.float32)

				start_time = time.time()
				# initw = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
				w2,dwdg2,dwdh2=self.mpqp.solve_QP(gw,Hw,prec=512)
				print('time elapsed',time.time() - start_time)
				# w2,dwdg2,dwdh2=self.mpqp.solve_QP(gw,Hw,initw = w2,prec=512)
				# print('time elapsed',time.time() - start_time)
				w2 = np.array(w2)[np.newaxis].T
				print('ground reaction force',np.dot(A,w2))


		else:
			#### uncomment this to have a continuous simulation....
			#self.q_dot_3D = self.C.value
			##TODO: Compute the Jocabian here
			dadx = self._contactFreeDynamics(C,D,u)
			dx_dotdx = np.zeros((30,42))
			dx_dotdx[0:15,15:30] = np.eye(15)
			dx_dotdx[15:30,:] = dadx
		#### uncomment this to have a continuous simulation....
		# self.q_3D = np.add(np.multiply(self.q_dot_3D,self.dt),self.q_3D)


		# self.robot.set_q_3D(self.q_3D)
		# self.robot.set_q_dot_3D(self.q_dot_3D)


		#################################################################
		# if NofContacts > 0:
		# 	return np.add(C,D@x_k)
		# else:
		# 	return C
		return

	def simulate(self,total_time,plot = False):
		world = self.robot.get_world()
		vis.add("world",world)
		vis.show()

		time.sleep(self.dt*200)

	
		#iteration_counter = 0
		
		u = [[9.92263016, -14.01755778,  -2.91153767, -30.94673869, 10.20010033,-0.22186062, \
			-11.28179175, -10.12199697,-4.59420828,  -3.32186255, -10.52210806,  -6.23841702],\
       		[-19.8456911 ,  -6.50670673, -14.23627271,  19.91256801,-11.36196078,  22.90960708,\
       		29.78007875,   6.52292582, -7.12403885, -39.70527496,  -1.65776911,  38.88391731],\
       		[-0.71478426,   0.21564621,  -0.15908892,   0.16485356, 0.57802428,   0.36248951,\
       		0.80722739,   0.11268208,-0.04014665,  -0.99691445,   0.85344531,   0.46476715]]

		iteration_counter = 0
		#while passed_time < total_time:
		while iteration_counter < 50:
			vis.lock()
			start = time.time()
			#self.simulateOnce(np.array(u[iteration_counter]))#,iteration_counter)
			#self.simulateOnce(u[iteration_counter%3])
			self.simulateOnce([6.08309021,0.81523653, 2.53641154, 5.83534863, 0.72158568, 2.59685143,\
				5.50487329, 0.54710471, 2.57836468 ,5.75260704 ,0.64075017, 2.51792186])
			iteration_counter += 1
			vis.unlock()
			time.sleep(self.dt*1.0)
		vis.kill()

	def _klamptFDRelated(self,C,D,wc,u):
		initial_q_3D = deepcopy(self.q_3D) #it is a column vector
		initial_q_dot_3D = deepcopy(self.q_dot_3D)
		#for dadx
		dadx = np.zeros((15,42))
		current_a = C + np.dot(D,wc)
		#debug
		current_a_3D = deepcopy(current_a)
		current_a = current_a[self.joint_indices_3D,:]
		#for dEdydx
		dQ1dx = np.zeros((self.Dx+self.Du,self.Dwc))
		current_x = self.robot.q_3D_to_2D(initial_q_3D)+self.robot.q_3D_to_2D(initial_q_dot_3D)
		C = np.add(np.multiply(C,self.dt),self.q_dot_3D) 
		D = np.multiply(D,self.dt)
		current_Q1 = np.dot(np.add(C,np.dot(D,wc)).T,np.dot(self.robot.get_mass_matrix(),D))
		
		eps = 1e-6

		for i in range(self.Dx+self.Du):
		#for i in range(1):
			if i < self.Dx:
				FD_add = [0]*self.Dx
				FD_add[i] = eps
				FD_x = vo.add(current_x,FD_add)
				self.robot.set_q_2D_(FD_x[0:int(self.Dx/2)])
				self.robot.set_q_dot_2D_(FD_x[int(self.Dx/2):self.Dx])

				C, D = self.robot.compute_CD(u)
			else:
				FD_x = current_x
				FD_add = [0]*self.Du
				FD_add[i-self.Dx] = eps
				FD_u = vo.add(u,FD_add)
				self.robot.set_q_2D_(current_x[0:int(self.Dx/2)])
				self.robot.set_q_dot_2D_(current_x[int(self.Dx/2):self.Dx])
				C, D = self.robot.compute_CD(FD_u)
			#for dadx
			a = C + np.dot(D,wc)
			a = a[self.joint_indices_3D,:]
			dadx[:,i] = np.multiply(np.subtract(a,current_a),1.0/eps).flatten()
			#for dEdydx
			C = np.add(np.multiply(C,self.dt),np.array(self.robot.q_2D_to_3D_(FD_x[int(self.Dx/2):self.Dx]))[np.newaxis].T) 
			D = np.multiply(D,self.dt)
			Q1 = np.dot(np.add(C,np.dot(D,wc)).T,np.dot(self.robot.get_mass_matrix(),D))
			dQ1dx[i,:] = np.multiply(np.subtract(Q1[0],current_Q1),1.0/eps)
		print('dadx',dadx[:,0:3])
		self.robot.set_q_3D(initial_q_3D)
		self.robot.set_q_dot_3D(initial_q_dot_3D)
		return dQ1dx,dadx

	def _contactFreeDynamics(self,C,D,u):
		initial_q_3D = deepcopy(self.q_3D) #it is a column vector
		initial_q_dot_3D = deepcopy(self.q_dot_3D)
		#for dadx
		dadx = np.zeros((15,42))
		current_a = C
		current_a = current_a[self.joint_indices_3D,:]
		current_x = self.robot.q_3D_to_2D(initial_q_3D)+self.robot.q_3D_to_2D(initial_q_dot_3D)
		C = np.add(np.multiply(C,self.dt),self.q_dot_3D) 
		D = np.multiply(D,self.dt)
		eps = 1e-6
		for i in range(self.Dx+self.Du):
			if i < self.Dx:
				FD_add = [0]*self.Dx
				FD_add[i] = eps
				FD_x = vo.add(current_x,FD_add)
				self.robot.set_q_2D_(FD_x[0:int(self.Dx/2)])
				self.robot.set_q_dot_2D_(FD_x[int(self.Dx/2):self.Dx])
				C, D = self.robot.compute_CD(u)
			else:
				FD_x = current_x
				FD_add = [0]*self.Du
				FD_add[i-self.Dx] = eps
				FD_u = vo.add(u,FD_add)
				self.robot.set_q_2D_(current_x)
				self.robot.set_q_dot_2D_(current_x)
				C, D = self.robot.compute_CD(FD_u)
			#for dadx
			a = C
			a = a[self.joint_indices_3D,:]
			dadx[:,i] = np.multiply(np.subtract(a,current_a),1.0/eps).flatten()


		self.robot.set_q_3D(initial_q_3D)
		self.robot.set_q_dot_3D(initial_q_dot_3D)
		return dadx


	def generateContacts(self):
		##How to generate contact on a curved surface needs a bit more thoughts/care
		"""Right not do not handle the case where a rigid body's angle is > pi/2.."""

		#print(self.robot.get_ankle_positions())

		contacts = []
		limb_indices = []
		NofContacts = 0

		#loop through the 4 ankles
		positions = self.robot.get_ankle_positions()
		for i in range(4):
			p = positions[i]

			##instead of terminating, just return the wrench space at the boundary		
			# if p[1] < self.terrain.material_range[0] or math.fabs(p[2]) > self.terrain.material_range[1]:
			# 	print('One of more ankles are penetrating the terrain outside of database range')
			# 	return
			flag = False 
			if p[1] < self.terrain.material_range[0]:
				#p[1] = self.terrain.material_range[0]
				flag= True
			if p[2] > self.terrain.material_range[1]:
				#p[2] = self.terrain.material_range[1]
				flag = True
			if p[2] <  -self.terrain.material_range[1]:
				#p[2] = -self.terrain.material_range[1]
				flag = True
			# if flag:
			# 	print('One of more ankles are penetrating the terrain outside of database range,using extrapolation')
			if p[1] <= 0:
				if p[2] >= 0:
					contact = [p[1],p[2],1,i,0] #the last element doesn't really mean anything, it's from the matlab program...
				else:
					contact = [p[1],-p[2],-1,i,0]
				contacts.append(contact)
				limb_indices.append(i)
				NofContacts += 1
		#print(contacts)
		return contacts,NofContacts,limb_indices

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

	def _3D_to_2D(self,q):
		q_2D = np.zeros(15)
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_2D[j] = q[i]
		return	q_2D

	def _unvectorize(self,v,NofRows,NofCols):
		Q = np.zeros((NofRows,NofCols))
		for i in range(NofCols):
			for j in range(NofRows):
				Q[j,i] = v[i*NofRows+j]

		return Q

	def _zero_3D_vector(self,x):
		for i in self.fixed_joint_indicies:
			x[i] = 0.0

if __name__=="__main__":
	# q_2D = np.array([0.0,1.1,0.1] + [0.3- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	#  	+[0.6+1.5708,0.0,-0.6])[np.newaxis] #tilted drop of robosimian

	# q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	#   	+[0.6+1.5708,0.0,-0.6])[np.newaxis] #four feet on the ground at the same time 

	### Embedded deep in granular surface
	# q_2D = np.array([0.0,0.936,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	#   	+[0.6+1.5708,0.0,-0.6])[np.newaxis] #four feet on the ground at the same time 

	q_2D = np.array([0.0,0.920,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	  	+[0.6+1.5708,0.0,-0.6])[np.newaxis] #four feet on the ground at the same time 
	q_dot_2D = np.array([0.0]*15)[np.newaxis]
	q_2D = q_2D.T
	q_dot_2D = q_dot_2D.T
	simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = 0.005, solver = 'cvxpy')
	#simulator.debugSimulation(20)
	u = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
		5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])
	simulator.simulateOnce(u)

	simulator = robosimianSimulator(q = q_2D,q_dot = q_dot_2D, dt = 0.005, solver = 'mpqp')
	#simulator.debugSimulation(20)
	u = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
		5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])
	simulator.simulateOnce(u)
	#simulator.simulate(1)
	#print(simulator._3D_to_2D(simulator.q_3D),simulator._3D_to_2D(simulator.q_dot_3D))
