#The 2D simulator traversing flat granular media terrain
#Use x-z 2D plane
#This file includes both MPQP and CVXPY as solvers
#only use the V formulation
from robosimian_utilities import granularMedia
from robosimian_wrapper import robosimian
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cvxpy as cp
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
	def __init__(self,q = np.zeros((15,1)), q_dot= np.zeros((15,1)) , dt = 0.01,dyn = 'own',print_level = 0, \
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
		if self.dyn == 'own':
			self.robot = robosimian(print_level = print_level, RL = RL)
			self.terrain = granularMedia(material = "sand",print_level = print_level, augmented = augmented, extrapolation = extrapolation)
			self.compute_pool = mp.Pool(4)
			#parameters for doing sensitivity analysis
			self.Dx = 30 
			self.Du = 12
			self.Dwc = 12 
			self.Dlambda = 4*26
			self.augmented = augmented
			self.robot.set_q_2D(self.q)
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
			#	cp.quad_form(self.Jp@(self.C+self.D@self.x[0:12]),self.M_2))
			self.obj = cp.Minimize(cp.quad_form(self.C+self.D@self.x[0:12],self.M))

			self.constraints = [self.x[0:12] - self.A@self.x[12:12+26*4]== np.zeros((12,1)),\
				self.A2@self.x[12:12+26*4] <= self.b2,\
				-self.x[12:12+26*4] <= np.zeros((26*4,1))]
			self.prob = cp.Problem(self.obj, self.constraints)
			A2 = np.zeros((4,self.Dlambda))
			for i in range(4):
				A2[i,i*26:(i+1)*26] = np.ones((1,26)) 	
			self.A2.value = A2		
			#self.expr = cp.transforms.indicator(self.constraints)

			#things for jacobian
			self.dEyy = np.zeros((self.Dwc+self.Dlambda,self.Dwc+self.Dlambda))
			self.dEyx = np.zeros((self.Dx+self.Du,self.Dwc+self.Dlambda))
			#dhdy
			self.dhy = np.zeros((self.Dwc+4+self.Dlambda,self.Dwc+self.Dlambda))
			self.dhy[0:self.Dwc,0:self.Dwc] = np.eye(self.Dwc)
			self.dhy[self.Dwc+4:self.Dwc+4+self.Dlambda,self.Dwc:self.Dwc+self.Dlambda] = -np.eye(self.Dlambda)
			for i in range(4):
				self.dhy[12+i,self.Dwc+i*26:self.Dwc+(i+1)*26] = np.ones((1,26))
			#ddh/dydy
			self.dhyy = np.zeros((self.Dwc+self.Dlambda,self.Dwc+self.Dlambda))
			#mu ddh/dydx
			self.mudhyx = np.zeros((self.Dx+self.Du,self.Dwc+self.Dlambda))
			self.dhx = np.zeros((120,self.Dx+self.Du))

		elif self.dyn == 'diffne':
			world = klampt.WorldModel()
			#TODO, change collision mesh size
			self.robot =  DiffNERobotModel(world,"Robosimian/robosimian_caesar_new_all_active.urdf",use2DBase=True)
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
		if self.dyn == 'own':
			q = x[0:15]
			q_dot = x[15:30]
			u = deepcopy(control)
		else:
			q = self._own_to_diffne(x[0:15])
			q_dot = self._own_to_diffne(x[15:30])
			u = -deepcopy(control)
			qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		if self.dyn == 'own':
			self.robot.set_q_2D_(q)
			self.robot.set_q_dot_2D_(q_dot)
			_,a, DynJac = self.simulateOnce(u, continuous_simulation = continuous, SA = True)
		else:
			tau = [0,0,0] + u.tolist() #diffNE has control on all dofs
			qdqNext,Dqdq,Dtau=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=True,Dtau=True)
			a = (np.array(qdqNext[15:30]) - qdq[15:30])/self.dt
			
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
			#calcualte DynJac
			Dqdq = np.array(Dqdq) #30-by-30
			Dtau = np.array(Dtau) #30-by-15
			#x means q, q_dot, tau
			#dq_dotdx = np.hstack((Dqdq[15:30,:],Dtau[15:30,3:15]))
			dq_dotdx = np.hstack((np.zeros((15,15)),np.eye(15),np.zeros((15,12))))

			dadq = Dqdq[15:30,0:15]/self.dt
			dadq_dot = (Dqdq[15:30,15:30] - np.eye(15))/self.dt 
			dadtau = Dtau[15:30,3:15]/self.dt
			dadx = np.hstack((dadq,dadq_dot,dadtau))

			DynJac = np.vstack((dq_dotdx,dadx))
			#now flip the axes back
			DynJac = self._own_to_diffne(J = DynJac)

		if not continuous:
			return a,DynJac
		else:
			if self.dyn == 'own':
				return a,DynJac,np.concatenate((self.q.ravel(),self.q_dot.ravel()))
			else:
				self.q = self._own_to_diffne(q = np.array(qdqNext[0:15]))[np.newaxis].T
				self.q_dot = self._own_to_diffne(q = np.array(qdqNext[15:30]))[np.newaxis].T
				return a,DynJac,np.concatenate((self.q.ravel(),self.q_dot.ravel()))

	def getDynJacNext(self,x,control):
		"""
		The dynamics jocabian here is not da/dx, but dxnext/dx

		"""
		if self.dyn == 'own':
			raise RuntimeError('own dynamics does not have this implemented yet')
		
		q = self._own_to_diffne(x[0:15])
		q_dot = self._own_to_diffne(x[15:30])
		u = -deepcopy(control)
		qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		tau = [0,0,0] + u.tolist() #diffNE has control on all dofs
		qdqNext,Dqdq,Dtau=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=True,Dtau=True)
		
		#calcualte DynJac
		Dqdq = np.array(Dqdq) #30-by-30
		Dtau = np.array(Dtau) #30-by-15
		#x means q, q_dot, tau
		DynJac = np.hstack((Dqdq,Dtau[:,3:15]))
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
		if self.dyn == 'own':
			q = x[0:15]
			q_dot = x[15:30]
			u = deepcopy(control)
		else:
			q = self._own_to_diffne(x[0:15])
			q_dot = self._own_to_diffne(x[15:30])
			u = -deepcopy(control)
			qdq = q.tolist() + q_dot.tolist()

		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		if self.dyn == 'own':
			self.robot.set_q_2D_(q)
			self.robot.set_q_dot_2D_(q_dot)
			force,a = self.simulateOnce(u, continuous_simulation = continuous)

		else:
			
			tau = [0,0,0] + u.tolist() #diffNE has control on all dofs

			qdqNext,_,_=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=False,Dtau=False)
			a = (np.array(qdqNext[15:30]) - np.array(qdq)[15:30])/self.dt
			
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
		if not continuous:
			return a
		else:
			if self.dyn == 'own':
				return a,np.concatenate((self.q.ravel(),self.q_dot.ravel()))
			else:
				self.q = self._own_to_diffne(q = np.array(qdqNext[0:15]))[np.newaxis].T
				self.q_dot = self._own_to_diffne(q = np.array(qdqNext[15:30]))[np.newaxis].T
				return a,np.concatenate((self.q.ravel(),self.q_dot.ravel()))

	def getStaticTorques(self,x):
		q = x[0:15]
		q_dot = x[15:30]
		self.q = q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T
		self.robot.set_q_2D_(q)
		self.robot.set_q_dot_2D_(q_dot)
		u = 0
		_,t = self.simulateOnce(u, fixed = True)
		return t

	def simulateOnceRL(self,u):
		_ = self.simulateOnce(u,True,False,False)

		return np.vstack((self.q,self.q_dot,self.ankle_poses))

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

		if self.dyn == 'own':
			if self.profile_computation:
				loop_start_time = time.time()
			##add viscious friction
			if self.RL:
				u_friction = []
				for i in range(12):
					u_friction.append(-configs.k_v*self.q_dot[i+3,0]**2)
				u_friction = np.array(u_friction)
			contacts,NofContacts,limb_indices = self.generateContacts()


			if self.print_level == 1:
				print('contacts',contacts)
			A = np.zeros((self.Dwc,self.Dlambda))

			b2 = np.zeros((4,1))
			Jp = self.robot.compute_Jp(contact_list=limb_indices)
			
			#  accel = C + D*wc, where wc is contact wrench
			if fixed:
				C,D,L_prime,L_J = self.robot.compute_CD_fixed()
			else:
				if self.RL:
					C, D = self.robot.compute_CD(u+u_friction)

				else:
					C, D = self.robot.compute_CD(u)
			
			if self.print_level == 1:
				print('C:')
				print(C)
			if SA:
				Q4s_all_limbs = []
				######compute the wrench spaces serially #####
				#SA
				self.dhy[0:12,12:116] = np.zeros((12,104))

			if self.profile_computation:
				print('Calculated Dynamics:',time.time() - loop_start_time)

			##### compute the wrench space of at 4 ankles in parallel using process ###
			args = []
			if NofContacts > 0:
				for i in range(4):
					if i <= NofContacts-1:
						args.append([contacts[i],self.robot.ankle_length,True])
					else:
						args.append([0,0,False])
				
				#compute_pool = MyPool(NofContacts)
				res = self.compute_pool.starmap(self.terrain.feasibleWrenchSpace,args)

				for i in range(NofContacts):
					add_A = res[i][0]
					Q4s = res[i][1]
					A[contacts[i][3]*3:(contacts[i][3]+1)*3,contacts[i][3]*26:(contacts[i][3]+1)*26] = add_A
					b2[contacts[i][3]] = 1
					if SA:
						self.dhy[contacts[i][3]*3:(contacts[i][3]+1)*3,contacts[i][3]*26+12:(contacts[i][3]+1)*26+12] = -add_A
						Q4s_all_limbs.append(Q4s)

			if self.profile_computation:
				print('Calculated Wrench space and drivatives:',time.time() - loop_start_time)

			# for contact in contacts:
			# 	add_A,Q4s = self.terrain.feasibleWrenchSpace(contact,self.robot.ankle_length,True)
			# 	A[contact[3]*3:(contact[3]+1)*3,contact[3]*26:(contact[3]+1)*26] = add_A
			# 	#SA
			# 	if SA:
			# 		self.dhy[contact[3]*3:(contact[3]+1)*3,contact[3]*26+12:(contact[3]+1)*26+12] = -add_A
			# 		Q4s_all_limbs.append(Q4s)
			# 	b2[contact[3]] = 1


			if self.print_level == 1:
				print("wrench vertices from limb 1",A[0:3,0:26])
				# print("active vertex",A[3:6,9+26])
			if NofContacts > 0:
				self.C.value = C*self.dt + self.q_dot
				self.D.value = np.multiply(D,self.dt)
				self.M.value = self.robot.get_mass_matrix()
				self.Jp.value = Jp
				self.A.value = A
				#self.A2.value = A2
				self.b2.value = b2

				# start_time = time.time()
				#smosek_param = {'MSK_DPAR_BASIS_TOL_X':1e-9,'MSK_DPAR_INTPNT_CO_TOL_MU_RED':1e-15,'MSK_DPAR_INTPNT_QO_TOL_MU_RED':1e-15,'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-12}
				#self.prob.solve(solver=cp.MOSEK,mosek_params = mosek_param,verbose = False,warm_start = False)#,eps_abs = 1e-11,eps_rel = 1e-11,max_iter = 100000000)				
				self.prob.solve(solver=cp.MOSEK,verbose = False,warm_start = False)#,eps_abs = 1e-11,eps_rel = 1e-11,max_iter = 100000000)
				#self.prob.solve(solver=cp.ECOS,verbose = False,warm_start = False,abstol = 1e-12,reltol = 1e-12)				
				#self.prob.solve(solver=cp.OSQP,verbose = False,warm_start = False)#,eps_abs = 1e-12,eps_rel = 1e-12,max_iter = 10000000)
				#self.prob.solve(verbose = False,warm_start = True)
				# print('CVX solving took:',time.time() - start_time)

				if self.profile_computation:
					print('CVXPY finished:',time.time() - loop_start_time)

				x_k = self.x.value[0:12]
				wc = x_k

				if self.RL:
					print('ground reaction force',x_k)
					print('ankle poses:',self.ankle_poses)

				if self.print_level == 1:
					print('ground reaction force',x_k)
					print('acceleration')
					print(C + D@x_k)
					print('objective value',(self.C.value+self.D.value@self.x.value[0:12]).T@self.M.value@(self.C.value+self.D.value@self.x.value[0:12]))

				if not self.prob.status == "optimal":
					print("cvxpy status:", self.prob.status)
			
				#debug:
				if self.print_level == 1:
					#print('active constraints (multipliers not close to  0:')
					gammas = np.vstack((self.constraints[1].dual_value,self.constraints[2].dual_value))

					print('constraint values:',self.A2.value@self.x.value[12:12+26*4])
					print(self.x.value[12:12+26*4])
					

				#Sensitivity analysis 
				if SA:
					self.dEyy[0:12,0:12] = self.D.value.T@self.M.value@self.D.value		
					if self.profile_computation:
						start_time_dyn = time.time()

					dQ1dx,dadx = self._klamptFDRelated(C,D,x_k,u)

					if self.profile_computation:
						print('Dynamics Derivative:',time.time() - start_time_dyn)
						
					self.dEyx[0:self.Dx+self.Du,0:12] = dQ1dx
					self.mudhyx[:,12:116] = np.zeros((self.Dx+self.Du,self.Dlambda))
					mus = self.constraints[0].dual_value#only need these mus
					lambdas = self.x.value[12:12+4*26]
					#todo: do not need to zero entire matrix
					self.dhx = np.zeros((120,self.Dx+self.Du))
					for (limb_index,Q4s,contact) in zip(limb_indices,Q4s_all_limbs,contacts):
						counter = 0
						Q5 = np.zeros((3,30))
						for Q4 in Q4s:
							#deal with the jocobian 
							J_raw = np.hstack((Jp[[limb_index*3+1,limb_index*3+2],:],np.zeros((2,15))))
							##TODO, smooth this out for the not augmented situation
							if not self.augmented:
								if contact[2] < 0:
									J_raw[1,:] = - J_raw[1,:]

							tmp = self._unvectorize(Q4,3,2)@J_raw						
							Q5 = Q5 - lambdas[limb_index*26+counter]*tmp



							Q4 = -mus[limb_index*3:limb_index*3+3,0]@tmp
							self.mudhyx[0:self.Dx,12+limb_index*26+counter] = Q4
							counter = counter + 1

					#Here:
					gammas = np.vstack((self.constraints[1].dual_value,self.constraints[2].dual_value))
					constraint_values = np.vstack((self.A2.value@self.x.value[12:12+26*4]-b2,-self.x.value[12:12+26*4]))

									
					dhy_bar = deepcopy(self.dhy)
					for i in range(self.Dlambda+4):
						dhy_bar[i+12,:] = dhy_bar[i+12,:]*gammas[i,0]

					F1 = np.zeros((self.Dwc + self.Dlambda + 120,self.Dwc + self.Dlambda + 120))
					F1[0:self.Dwc + self.Dlambda,0:self.Dwc + self.Dlambda] = self.dEyy.T
					F1[0:self.Dwc + self.Dlambda,self.Dwc + self.Dlambda:self.Dwc + self.Dlambda+120] = self.dhy.T
					F1[self.Dwc + self.Dlambda:self.Dwc + self.Dlambda+120,0:self.Dwc + self.Dlambda] = dhy_bar
					#Here
					G = np.eye(self.Dlambda+4)
					for i in range(self.Dlambda+4):
						G[i,i] = constraint_values[i,0]
					for i in range(len(gammas)):
						self.dhx[12+i,:] = gammas[i,0]*self.dhx[12+i,:]

					#Here
					F1[self.Dwc + self.Dlambda+12:self.Dwc + self.Dlambda+120,self.Dwc + self.Dlambda+12:self.Dwc + self.Dlambda+120] = G
					F2 = np.vstack((-self.dEyx.T-self.mudhyx.T,-self.dhx)) #self.dhx already contains the negative?
					##
					dymu_dx = np.linalg.inv(F1)@F2
					dwc_dx = dymu_dx[0:self.Dwc,:]
					if self.print_level == 1:
						print('dwcdx',dwc_dx[:,0:3])
					dadx_full = D@dwc_dx + dadx
					dx_dotdx = np.zeros((30,42))
					dx_dotdx[0:15,15:30] = np.eye(15)
					dx_dotdx[15:30,:] = dadx_full

					if self.profile_computation:
						print('SA finished:',time.time() - loop_start_time)

				if fixed:
					all_fixed_joint_torques = np.add(L_prime,(L_J@x_k).ravel())
					fixed_joint_torques = all_fixed_joint_torques[[5,7,9,13,15,17,21,23,25,29,31,33]]
					if self.print_level == 1:
						print('fixed joint torques',fixed_joint_torques)
				if continuous_simulation:
					self.q_dot = C*self.integrate_dt + self.q_dot+D@wc*self.integrate_dt

			else:
				#### uncomment this to have a continuous simulation....
				if continuous_simulation:
					self.q_dot = C*self.integrate_dt + self.q_dot
				##TODO: Compute the Jocabian here
				if SA:
					dadx = self._contactFreeDynamics(C,D,u)
					dx_dotdx = np.zeros((30,42))
					dx_dotdx[0:15,15:30] = np.eye(15)
					dx_dotdx[15:30,:] = dadx
			#### uncomment this to have a continuous simulation....
			
			if continuous_simulation:
				self.q = self.q_dot*self.integrate_dt+self.q
				self.robot.set_q_2D_(self.q.ravel().tolist())
				self.robot.set_q_dot_2D_(self.q_dot.ravel().tolist())
				if self.print_level == 1:
					print('current q:',self.q)

				if self.RL:
					self.q = self._check_q_bounds(self.q)
					self.q_dot = self._check_q_dot_bounds(self.q_dot)

				self.robot.set_q_2D(self.q)
				self.robot.set_q_dot_2D(self.q_dot)

			#################################################################
			if fixed:
				if NofContacts > 0:
					return C+D@wc,fixed_joint_torques
				else:
					return C,[]

			if not self.RL:
				if NofContacts > 0:
					if SA:
						return x_k,C+D@wc,dx_dotdx
					else:
						return x_k,C+D@wc
				else:
					pass
			else:
				##what do you return when you do RL?
				return

		elif self.dyn == 'diffne':

			q = self._own_to_diffne(q = self.q.ravel())
			q_dot = self._own_to_diffne(q = self.q_dot.ravel())
			u1 = -deepcopy(u)
			qdq = q.tolist() + q_dot.tolist()
			tau = [0,0,0] + u1.tolist() #diffNE has control on all dofs

			qdqNext,_,_=self.robot.simulate(self.dt,qdq,tau=tau,Dqdq=False,Dtau=False)
			a = (np.array(qdqNext[15:30]) - np.array(qdq)[15:30])/self.dt
			#flip the axes back
			a = self._own_to_diffne(q = a)
			a = a[np.newaxis].T
			if not continuous_simulation:
				return 
			else:
				if self.integration == 'semi-Euler':
					self.q = self._own_to_diffne(q = np.array(qdqNext[0:15]))[np.newaxis].T
					self.q_dot = self._own_to_diffne(q = np.array(qdqNext[15:30]))[np.newaxis].T
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

	# def debugSimulationTraj(self,qs,dt = 0.1):
	# 	"""
	# 	qs: a list of configurations
	# 	"""

	# 	world = self.robot.get_world()
	# 	vis.add("world",world)
	# 	vis.show()

	# 	vis.show()
	# 	for q in qs:
	# 		vis.lock()
	# 		self.reset(np.array(q))
	# 		vis.unlock()
	# 		time.sleep(dt)

	# 	while vis.shown():
	# 		time.sleep(dt)
	# 	vis.kill()

	def reset(self,q,q_dot = np.array([0]*15)):
		"""
		Parameters:
		numpy 1D array
		"""
		self.q= q[np.newaxis].T
		self.q_dot = q_dot[np.newaxis].T

		if self.dyn == 'own':
			self.robot.set_q_2D_(q)
			self.robot.set_q_dot_2D_(q_dot)

		if self.RL:
			_ = self.generateContacts()
			return np.concatenate((q,q_dot,np.ravel(self.ankle_poses)))
		else:
			return 
	def _klamptFDRelated(self,C,D,wc,u):
		initial_q = deepcopy(self.q) #it is a column vector
		initial_q_dot = deepcopy(self.q_dot)
		#for dadx
		dadx = np.zeros((15,42))
		current_a = C + D@wc

		#for dEdydx
		dQ1dx = np.zeros((self.Dx+self.Du,self.Dwc))
		current_x = np.ravel(np.vstack((initial_q,initial_q_dot)))
		C = (C*self.dt)+self.q_dot
		D = D*self.dt
		current_Q1 = (C+D@wc).T@(self.robot.get_mass_matrix()@D)
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
			#a = a[self.joint_indices_3D,:]
			dadx[:,i] = np.multiply(np.subtract(a,current_a),1.0/eps).flatten()
			#for dEdydx
			C = np.add(np.multiply(C,self.dt),np.array(FD_x[int(self.Dx/2):self.Dx])[np.newaxis].T) 
			D = np.multiply(D,self.dt)
			Q1 = np.dot(np.add(C,np.dot(D,wc)).T,np.dot(self.robot.get_mass_matrix(),D))
			dQ1dx[i,:] = np.multiply(np.subtract(Q1[0],current_Q1),1.0/eps)
		#print('dadx',dadx[:,0:3])
		self.robot.set_q_2D(initial_q)
		self.robot.set_q_dot_2D(initial_q_dot)
		return dQ1dx,dadx

	def _contactFreeDynamics(self,C,D,u):
		initial_q = deepcopy(self.q) #it is a column vector
		initial_q_dot = deepcopy(self.q_dot)
		#for dadx
		dadx = np.zeros((15,42))
		current_a = C
		current_x = np.ravel(np.vstack((initial_q,initial_q_dot)))
		C = np.add(np.multiply(C,self.dt),self.q_dot) 
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
			dadx[:,i] = np.multiply(np.subtract(a,current_a),1.0/eps).flatten()


		self.robot.set_q_2D(initial_q)
		self.robot.set_q_dot_2D(initial_q_dot)
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
			if self.RL:
				self.ankle_poses[i*2,0] = p[1]
				self.ankle_poses[i*2+1,0] = p[2]
			# flag = False 
			# if p[1] < self.terrain.material_range[0]:
			# 	#p[1] = self.terrain.material_range[0]
			# 	flag= True
			# if p[2] > self.terrain.material_range[1]:
			# 	#p[2] = self.terrain.material_range[1]
			# 	flag = True
			# if p[2] <  -self.terrain.material_range[1]:
			# 	#p[2] = -self.terrain.material_range[1]
			# 	flag = True
			# if flag:
			# 	print('One of more ankles are penetrating the terrain outside of database range,using extrapolation')
			#if p[1] <= 0:
			#even of not contact, still give a contact force 
			if p[2] >= 0:
				contact = [p[1],p[2],1,i,0] #the last element doesn't really mean anything, it's from the matlab program...
			else:
				if not self.augmented:
					contact = [p[1],-p[2],-1,i,0]
				else:
					contact = [p[1],p[2],1,i,0]
			if self.augmented:
				contacts.append(contact)
				limb_indices.append(i)
				NofContacts += 1
			else:
				if p[1] <= 0:
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
		indeces = [1] + [3,4,5,6,7,8,9,10,11,12,13,14] #+ [16] + [18,19,20,21,22,23,24,25,26,27,28,29]
		if q is not None:		
			q_new = deepcopy(q)
			indeces = [1] + [3,4,5,6,7,8,9,10,11,12,13,14]
			for i in indeces:
				q_new[i] = -q_new[i]
			return q_new
		elif J is not None:
			J_new = deepcopy(J)

			indeces_row =  [1] + [3,4,5,6,7,8,9,10,11,12,13,14] + [16] + [18,19,20,21,22,23,24,25,26,27,28,29]
			for i in indeces_row:
				#each column corresponds to each dof
				J_new[i,:] = -J_new[i,:]

			indeces_col =  [1] + [3,4,5,6,7,8,9,10,11,12,13,14] + [16] + [18,19,20,21,22,23,24,25,26,27,28,29] + [30,31,32,33,34,35,36,37,38,39,40,41]
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

	def closePool(self):
		self.compute_pool.close()

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