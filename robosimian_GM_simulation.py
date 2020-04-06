#The 2D simulator traversing flat granular media terrain
#Use x-z 2D plane
from robosimian_utilities import granularMedia
from robosimian_model_klampt import robosimian
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
import sympy

class robosimianSimulator:
	def __init__(self,q_2D = None, q_dot_2D = None , dt = 0.01, formulation = 'H', solver = 'cvxpy'):
		if q_2D.any():
			self.q_2D = q_2D
		else:
			self.q_2D=np.array([0.0,0.7,0.0] + [0.0]*12)[np.newaxis]
			self.q_2D = self.q_2D.T

		if q_dot_2D.any():
			self.q_dot_2D = q_dot_2D
		else:
			self.q_dot_2D = np.array([0.0]*15)[np.newaxis]
			self.q_dot_2D = self.q_dot_2D.T
		self.robot = robosimian()
		self.q_3D = self.robot.set_q_2D(self.q_2D)
		self.q_dot_3D = self.robot.set_q_dot_2D(self.q_dot_2D)
		self.terrain = granularMedia(material = "sand")

		self.joint_indices_3D = [0,2,4,8,10,12,16,18,20,24,26,28,32,34,36]
		self.fixed_joint_indicies = [1,3,5,6,7,9,11,13,14,15,17,19,21,22,23,25,27,29,30,31,33,35,37]
		self.joint_indices_2D = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
		self.timer = [0,0,0] #time spent at 
		self.time = 0
		self.dt = dt
		self.formulation = formulation
		self.damping = 0.0 #maybe include this for stability
		self.Dx = 30
		self.Du = 12
		self.Dwc = 12
		self.Dlambda = 4*26
		self.NofJoints3D = 38
		self.solver = solver
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

		#PINOCCHIO
		

		##using half-space or vertex representation of the wrench space
		#vertex representation is used for sensitivity analysis
		if formulation == 'H':	
			self.estimated_upper_bound = 50  ## we don't the exact number of constraints 
			##set up parameters for cvxpy to warm start the problem...
			self.C = cp.Parameter((38,1)) #joint constraints
			self.D = cp.Parameter((38,12)) #contact jacobian
			self.M = cp.Parameter((38,38),PSD=True)
			self.Jp = cp.Parameter((12,38))
			self.Ju = cp.Parameter((39,12))
			self.A = cp.Parameter((self.estimated_upper_bound*4,12)) #contact wrench..
			self.b = cp.Parameter((self.estimated_upper_bound*4,1))
			self.Aeq = cp.Parameter((12,12))
			self.beq = cp.Parameter((12,1))
			self.x = cp.Variable((4*3,1))
			self.obj = cp.Minimize(cp.quad_form(self.C+self.D@self.x,self.M)+\
				cp.quad_form(self.Jp*(self.C+self.D@self.x),self.M_2))
			self.constraints = [self.A@self.x <= self.b,self.Aeq@self.x == self.beq]
			
			self.prob = cp.Problem(self.obj, self.constraints)			

		elif formulation == "V":
			if self.solver == 'cvxpy':
				self.C = cp.Parameter((38,1)) #joint constraints
				self.D = cp.Parameter((38,12)) #contact jacobian
				self.M = cp.Parameter((38,38),PSD=True)
				self.Jp = cp.Parameter((12,38))
				self.Ju = cp.Parameter((39,12))
				#for each contact there are 26 lambdas
				self.x = cp.Variable((4*3+4*26,1)) #contact wrench and lambdas
				self.A = cp.Parameter((3*4,26*4))
				self.A2 = cp.Parameter((4,26*4))
				self.b2 = cp.Parameter((4,1))
				self.obj = cp.Minimize(cp.quad_form(self.C+self.D@self.x[0:12],self.M)+\
					cp.quad_form(self.Jp*(self.C+self.D@self.x[0:12]),self.M_2))
				self.constraints = [self.x[0:12] - self.A@self.x[12:12+26*4] == np.zeros((12,1)),\
					self.A2@self.x[12:12+26*4] == self.b2,\
					-self.x[12:12+26*4] <= np.zeros((26*4,1))]
				self.prob = cp.Problem(self.obj, self.constraints)			
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

				self.filler_list = []
				for i in range(132):
					self.filler_list.append(i)
			elif self.solver == 'mpqp':
				#TODO:add mpqp
				pass
		else:
			print("Wrong formulation specification")
			exit()
		


	def debugSimulation(self,duration = 5):
		world = self.robot.get_world()
		vis.add("world",world)
		vis.show()


		startTime = time.time()
		u = np.zeros((12,1))

		while vis.shown():
			time.sleep(0.1)

		# while (time.time()-startTime < duration):
		# 	vis.lock()
		# 	#C,D = self.robot.compute_CD(u)
		# 	C,D = self.robot.compute_CD_fixed((0,0,-9.81)) ## all fixed....
		# 	print(C[0:6])
		# 	f = np.array([0,0,0,0,0,0,0,0,0,0,0,0])[np.newaxis].T
		# 	self.q_dot_3D = np.add(np.add(np.multiply(C,self.dt),self.q_dot_3D), np.multiply(D,self.dt)@f)
		# 	self.q_3D = np.add(np.multiply(self.q_dot_3D,self.dt),self.q_3D)
		# 	self.robot.set_q_3D(self.q_3D)
		# 	self.robot.set_q_dot_3D(self.q_dot_3D)
		# 	print(self.q_3D[0:6,0])
		# 	vis.unlock()
		# 	time.sleep(self.dt*5.0)
		vis.kill()

	def getDynamics(self,x,u):
		"""
		Parameters:
		----------
		Both x and u are 1D numpy arrays
		"""
		q_2D = x[0:15]
		q_dot_2D = x[15:30]
		print(q_2D,q_dot_2D)
		self.q_3D = self.robot.set_q_2D_(q_2D)
		self.q_dot_3D = self.robot.set_q_dot_2D_(q_dot_2D)
		q_dot_dot_3D,_ = self.simulateOnce(u) #it is a (n,1) np array

		return self._3D_to_2D(q_dot_dot_3D) #this becomes a numpyt 1D array

	def getDynJac(self,x,u):
		q_2D = x[0:15]
		q_dot_2D = x[15:30]
		self.q_3D = self.robot.set_q_2D_(q_2D)
		self.q_dot_3D = self.robot.set_q_dot_2D_(q_dot_2D)
		a,DynJac = self.simulateOnce(u)
		return self._3D_to_2D(a),DynJac

	def simulateOnce(self,u):#debug,counter):
		contacts,NofContacts,limb_indices = self.generateContacts()
		# print(contacts)
		#print('N of contacts:',NofContacts)
		### H&V representation 
		if self.formulation == 'H':
			#note that u here is a 1D numpy array
			#######toggle between fixed and free joints########
			C, D = self.robot.compute_CD(u)
			#C, D, L_prime, L_J = self.robot.compute_CD_fixed((0,0,-9.81))

			A = np.zeros((self.estimated_upper_bound*4,12)) 
			b = np.zeros((self.estimated_upper_bound*4,1))
			Aeq = np.eye(12)
			Jp = self.robot.compute_Jp(contact_list=limb_indices) #return all zeros..

			######compute the wrench spaces serially #####
			for contact in contacts:
				add_A,add_b = self.terrain.feasibleWrenchSpace(contact,self.robot.ankle_length,True,self.formulation)
				A[contact[3]*self.estimated_upper_bound:(contact[3]+1)*self.estimated_upper_bound,\
					contact[3]*3:(contact[3]+1)*3] = add_A
				b[contact[3]*self.estimated_upper_bound:(contact[3]+1)*self.estimated_upper_bound,:] = \
					add_b
				Aeq[contact[3]*3:contact[3]*3+3,contact[3]*3:contact[3]*3+3] = np.zeros((3,3))

			##### compute the wrench space of at 4 ankles in parallel using process ###
			'''
			args = []
			if NofContacts > 0:
				for i in range(4):
					if i <= NofContacts -1:
						args.append([contacts[i],self.robot.ankle_length,True])
					else:
						args.append([0,0,False])
				compute_pool = mp.Pool(NofContacts)
				res = compute_pool.starmap(self.terrain.feasibleWrenchSpace,args)

				for i in range(NofContacts):
					add_A = res[i][0]
					add_b = res[i][1]
					A[contacts[i][3]*self.estimated_upper_bound:(contacts[i][3]+1)*self.estimated_upper_bound,\
						contacts[i][3]*3:(contacts[i][3]+1)*3] = add_A
					b[contacts[i][3]*self.estimated_upper_bound:(contacts[i][3]+1)*self.estimated_upper_bound,:] = \
						add_b
					Aeq[contacts[i][3]*3:contacts[i][3]*3+3,contacts[i][3]*3:contacts[i][3]*3+3] = np.zeros((3,3))

			Timer
			WS_time = time.time() - last_time
			last_time = time.time()
			'''
			self.C.value = np.add(np.multiply(C,self.dt),self.q_dot_3D) #self.v is the full 38-long vector
			if NofContacts > 0:				
				self.D.value = np.multiply(D,self.dt)
				self.M.value = self.robot.get_mass_matrix()
				#print("M2 psd:",np.all(np.linalg.eigvals(self.M_2) >= 0))
				self.Jp.value = Jp
				self.A.value = A
				self.b.value = b
				self.Aeq.value = Aeq
				self.beq.value = np.zeros((12,1))
				#start_time = time.time()
				self.prob.solve(solver=cp.SCS,max_iters = 1000,verbose = False,warm_start = False)
				#self.prob.solve(solver=cp.OSQP,verbose = True,warm_start = False)
				#self.prob.solve(verbose = False,warm_start = True)
				#print("Elapsed time:",time.time() -start_time)
				#print("status:", self.prob.status)
				# print("optimal value", self.prob.value)
				# print("optimal var", self.x.value)
				#print(self.x.value[24:36])

				x_k = self.x.value
				print("status:", self.prob.status)
				print("The Lagrange multipliers are", self.constraints[0].dual_value)
				print("The Lagrange multipliers are", self.constraints[1].dual_value)
				#print(self.q_3D,self.q_dot_3D)
				self.q_dot_3D = np.add(self.C.value,self.D.value@x_k)

				#timer
				#opt_time = time.time() - last_time
				#self.timer = vo.add(self.timer,[dynamics_time,WS_time,opt_time])


				##### display the joint torques when fixed
				# all_fixed_joint_torques = np.add(L_prime,(L_J@x_k).ravel())
				# fixed_joint_torques = all_fixed_joint_torques[[5,7,9,13,15,17,21,23,25,29,31,33]]
				# #print(all_fixed_joint_torques)
			else:
				#print(self.C.value)
				self.q_dot_3D = self.C.value
				
				##### display the joint torques when fixed
				# all_fixed_joint_torques = L_prime
				# fixed_joint_torques = all_fixed_joint_torques[[5,7,9,13,15,17,21,23,25,29,31,33]]
			##### display the joint torques when fixed
			# print("joint torques:",fixed_joint_torques)

			#### uncomment this to have a continuous simulation....
			self.q_3D = np.add(np.multiply(self.q_dot_3D,self.dt),self.q_3D)
			self.robot.set_q_3D(self.q_3D)
			self.robot.set_q_dot_3D(self.q_dot_3D)
			
			#print(self.timer)
		elif self.formulation == 'V':

			A = np.zeros((self.Dwc,self.Dlambda))
			A2 = np.zeros((4,self.Dlambda)) 
			b2 = np.zeros((4,1))
			Jp,Jp_2D = self.robot.compute_Jp_2(contact_list=limb_indices)
			
			#gradient version..
			C, D = self.robot.compute_CD(u) # different from above, C = C*dt + Vk, D = D*dt

			Q4s_all_limbs = []
			######compute the wrench spaces serially #####
			#SA
			self.dhy[0:12,12:116] = np.zeros((12,104))
			#print('dhy:',self.dhy[12:16,:])
			for contact in contacts:
				add_A,Q4s = self.terrain.feasibleWrenchSpace(contact,self.robot.ankle_length,True,self.formulation)
				A[contact[3]*3:(contact[3]+1)*3,contact[3]*26:(contact[3]+1)*26] = add_A
				#SA
				self.dhy[contact[3]*3:(contact[3]+1)*3,contact[3]*26+12:(contact[3]+1)*26+12] = -add_A

				Q4s_all_limbs.append(Q4s)
				A2[contact[3],contact[3]*26:(contact[3]+1)*26] = np.ones((1,26))
				b2[contact[3]] = 1

			self.C.value = np.add(np.multiply(C,self.dt),self.q_dot_3D) 
			if NofContacts > 0:
				self.D.value = np.multiply(D,self.dt)
				self.M.value = self.robot.get_mass_matrix()
				self.Jp.value = Jp
				self.A.value = A
				self.A2.value = A2
				self.b2.value = b2

				start_time = time.time()
				#self.prob.solve(solver=cp.ECOS,max_iters = 100000,verbose = False,warm_start = False,abstol = 1e-16,reltol = 1e-16)
				#tmp1 = self.constraints[2].dual_value
				#print(tmp1)
				
				self.prob.solve(solver=cp.OSQP,verbose = False,warm_start = False,eps_abs = 1e-10,eps_rel = 1e-10,max_iter = 10000000)
				#self.prob.solve(solver=cp.OSQP,verbose = False,warm_start = False)#,eps_abs = 1e-12,eps_rel = 1e-12,max_iter = 10000000)
				#self.prob.solve(verbose = False,warm_start = True)
				#print(self.constraints[2].dual_value)
				#print('time:',time.time() - start_time)
				x_k = self.x.value[0:12]
				print('x_k',x_k)
				print("status:", self.prob.status)
				self.q_dot_3D = np.add(self.C.value,self.D.value@x_k)
				#set the fixed joint to 0
				self._zero_3D_vector(self.q_dot_3D)
				# start_time = time.time()
				##Calculate the jocobian...
				#-dEyy
				self.dEyy[0:12,0:12] = self.D.value.T@self.M.value@self.D.value
				# print('dEyy:',self.dEyy)
				# print('condition number of dEyy:',np.linalg.cond(self.dEyy))
				#-dQ1dx
				# start_time3 = time.time()
				dQ1dx,dadx = self._klamptFDRelated(C,D,x_k,u)
				# print('FD Took time:',time.time() - start_time3)
				self.dEyx[0:self.Dx+self.Du,0:12] = dQ1dx
				#-dhy calculated above
				#-dhyy is always zero
				#-mudhyx & dhdx:
				self.mudhyx[:,12:116] = np.zeros((self.Dx+self.Du,self.Dlambda))
				mus = self.constraints[0].dual_value#only need these mus
				lambdas = self.x.value[12:12+4*26]
				#todo: do not need to zero entire matrix
				self.dhx  = np.zeros((120,self.Dx+self.Du))
				for (limb_index,Q4s,contact) in zip(limb_indices,Q4s_all_limbs,contacts):
					counter = 0
					Q5 = np.zeros((3,30))
					for Q4 in Q4s:
						#deal with the jocobian 
						J_raw = np.hstack((Jp_2D[[limb_index*3+1,limb_index*3+2],:],np.zeros((2,15))))

						##TODO, smooth this out...
						if contact[2] < 0:
							J_raw[1,:] = - J_raw[1,:]
						tmp = self._unvectorize(Q4,3,2)@J_raw
						Q5 = Q5 - lambdas[limb_index*26+counter]*tmp
						Q4 = -mus[limb_index*3:limb_index*3+3,0]@tmp
						self.mudhyx[0:self.Dx,12+limb_index*26+counter] = Q4
						counter = counter + 1
					self.dhx[limb_index*3:limb_index*3+3,0:self.Dx] = Q5

				gammas = self.constraints[2].dual_value
				constraint_values = -self.x.value[12:12+26*4]


				dhy_bar = deepcopy(self.dhy)
				##debug
				for i in range(self.Dlambda):
					dhy_bar[i+12+4,:] = dhy_bar[i+12+4,:]*gammas[i,0]

				F1 = np.zeros((self.Dwc + self.Dlambda + 120,self.Dwc + self.Dlambda + 120))
				F1[0:self.Dwc + self.Dlambda,0:self.Dwc + self.Dlambda] = self.dEyy.T
				F1[0:self.Dwc + self.Dlambda,self.Dwc + self.Dlambda:self.Dwc + self.Dlambda+120] = self.dhy.T
				F1[self.Dwc + self.Dlambda:self.Dwc + self.Dlambda+120,0:self.Dwc + self.Dlambda] = dhy_bar
				G = np.eye(self.Dlambda)
				for i in range(self.Dlambda):
					G[i,i] = constraint_values[i,0]
				for i in range(len(gammas)):
					self.dhx[16+i,:] = gammas[i,0]*self.dhx[12+i,:]

				F1[self.Dwc + self.Dlambda+12+4:self.Dwc + self.Dlambda+120,self.Dwc + self.Dlambda+12+4:self.Dwc + self.Dlambda+120] = G
				F2 = np.vstack((-self.dEyx.T-self.mudhyx.T,-self.dhx)) #self.dhx already contains the negative?
				# print('Shape of F1 is',np.shape(F1))
				# print('Rank of F1 is',np.linalg.matrix_rank(F1))
				# print(self.dhy.T[110:116,0:12])
				# np.savetxt('quantity of interest',F1[110:116,:])

				# print('Rank is', np.linalg.matrix_rank(F1[111:236,:]))
				# print('Rank is', np.linalg.matrix_rank(F1[0:116,:]))
				#print('Rank should be',132+104-110)
				#print(self.filler_list+active_list)
				#print('Rank of F1 is',np.linalg.matrix_rank(F1))

				# F1 = F1[self.filler_list+active_list,:]
				# F1 = F1[:,self.filler_list+active_list]
				# print('Shape of F1 is',np.shape(F1))
				print('Rank of active constraint_values:',np.linalg.matrix_rank(F1[0:132,0:132]))
				w,v = np.linalg.eig(F1)
				print('Condition number of F1 is',np.min(np.absolute(w)))
				# print('Rank is',np.linalg.matrix_rank(F1[12:116,132:236]))
				#print(np.linalg.matrix_rank(F1[132:155,:]))
				# print(np.shape(F1))
				# print(F1[132:154,0:5])
				#print('Rank of F1 is',np.linalg.matrix_rank(F1))
				# start_time2 = time.time()

				##
				dymu_dx = np.linalg.inv(F1)@F2
				# print('Inverting the matrix took:',time.time() -start_time2)
				dwc_dx = dymu_dx[0:self.Dwc,:]
				print('dwc_dx',dwc_dx[:,0:2])
				#print(dwc_dx)
				# print('In total took time:',time.time() - start_time)

				#debug
				#dwc_dx = dwc_dx*0
				print('Debug:',F1[15,0:12])
				print(F2[0:12,15])

				### now calculate dx_dotdx
				D_2D = D[self.joint_indices_3D,:]
				#print('dadx',dadx[:,0:2])
				dadx_full = D_2D@dwc_dx + dadx
				dx_dotdx = np.zeros((30,42))
				dx_dotdx[0:15,15:30] = np.eye(15)
				dx_dotdx[15:30,:] = dadx_full
				self.q_3D = np.add(np.multiply(self.q_dot_3D,self.dt),self.q_3D)
				self._zero_3D_vector(self.q_3D)
				self.robot.set_q_3D(self.q_3D)
				self.robot.set_q_dot_3D(self.q_dot_3D)
			else:
				self.q_dot_3D = self.C.value
				##TODO: Compute the Jocabian here
				dadx = self._contactFreeDynamics(C,D,u)
				dx_dotdx = np.zeros((30,42))
				dx_dotdx[0:15,15:30] = np.eye(15)
				dx_dotdx[15:30,:] = dadx
			#### uncomment this to have a continuous simulation....
			self.q_3D = np.add(np.multiply(self.q_dot_3D,self.dt),self.q_3D)


			self.robot.set_q_3D(self.q_3D)
			self.robot.set_q_dot_3D(self.q_dot_3D)


		#################################################################
		if NofContacts > 0:
			return np.add(C,D@x_k),dx_dotdx
		else:
			return C,dx_dotdx

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
		current_a = C + D@wc
		#debug
		current_a_3D = deepcopy(current_a)
		current_a = current_a[self.joint_indices_3D,:]
		#for dEdydx
		dQ1dx = np.zeros((self.Dx+self.Du,self.Dwc))
		current_x = self.robot.q_3D_to_2D(initial_q_3D)+self.robot.q_3D_to_2D(initial_q_dot_3D)
		C = np.add(np.multiply(C,self.dt),self.q_dot_3D) 
		D = np.multiply(D,self.dt)
		current_Q1 = np.add(C,D@wc).T@self.robot.get_mass_matrix()@D
		
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
			a = C + D@wc
			a = a[self.joint_indices_3D,:]
			dadx[:,i] = np.multiply(np.subtract(a,current_a),1.0/eps).flatten()
			#for dEdydx
			C = np.add(np.multiply(C,self.dt),np.array(self.robot.q_2D_to_3D_(FD_x[int(self.Dx/2):self.Dx]))[np.newaxis].T) 
			D = np.multiply(D,self.dt)
			Q1 = np.add(C,D@wc).T@self.robot.get_mass_matrix()@D
			dQ1dx[i,:] = np.multiply(np.subtract(Q1[0],current_Q1),1.0/eps)
		
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

	q_2D = np.array([0.0,0.942,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
	  	+[0.6+1.5708,0.0,-0.6])[np.newaxis] #four feet on the ground at the same time 
	q_dot_2D = np.array([0.0]*15)[np.newaxis]
	q_2D = q_2D.T
	q_dot_2D = q_dot_2D.T
	simulator = robosimianSimulator(q_2D = q_2D,q_dot_2D = q_dot_2D, dt = 0.005, formulation = 'V')
	#simulator.debugSimulation(20)
	u = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])
	simulator.simulateOnce(u)
	#simulator.simulate(1)
	#print(simulator._3D_to_2D(simulator.q_3D),simulator._3D_to_2D(simulator.q_dot_3D))
