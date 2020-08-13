#klampt wrappper
from klampt import WorldModel
from klampt.math import vectorops as vo
from klampt.model import ik, collide
import numpy as np
import math

class robosimian:
	def __init__(self,dt = 0.01,print_level = 0, RL = False):
		self.world = WorldModel()
		####The 2D version has all but 12 joints fixed....
		self.world.loadElement("data/robosimian_caesar_new_2D.urdf")
		self.robot = self.world.robot(0)  ##Is this robot actually used????

		self.world_all_active = WorldModel()
		self.world_all_active.readFile("data/robosimian_2D_all_active_world.xml") 
		self.robot_all_active = self.world_all_active.robot(0)
		self.fixed_joint_indicies = [1,3,5,6,7,8,9,11,13,14,15,16,17,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
		#below is used for calculating static joint torques
		self.all_fixed_joint_indicies = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,\
			30,31,32,33,34,35,36,37]#this is for a completely fixed 2D robot, except for the torso
		self.active_joint_indices = [10,12,18,20] #just the limb, without the torso
		self.joint_indices_3D = [0,2,4,10,12,18,20]
		self.joint_indices_2D = [0,1,2,3,4,5,6]
		self.revolute_joint_indices_3D = [10,12,18,20]
		self.revolute_joint_indices_2D = [0,1,2,3]
		self.N_of_joints_3D = 38
		#self.ankle_length = 0.153
		# ####link 7-13 RF; 15-21 RR; 23-29 LR; 31-37 LF
		self.dt = dt
		self.print_level = print_level


		# desired_total_mass = 99.888 #99.888 is the original total mass
		# total_mass = 0.0
		# #link_list = [5] + list(range(7,14)) + list(range(15,22)) + list(range(23,30)) + list(range(31,37))

		# for i in range(self.N_of_joints_3D):
		# 	total_mass +=self.robot_all_active.link(i).getMass().getMass()
		# for i in range(self.N_of_joints_3D):
		# 	#print('---------')
		# 	#print(robot.link(i).getMass().getMass())
		# 	mass_to_be_set = self.robot_all_active.link(i).getMass().getMass() * desired_total_mass/total_mass
		# 	mass_structure = self.robot_all_active.link(i).getMass()
		# 	mass_structure.setMass(mass_to_be_set)
		# 	self.robot_all_active.link(i).setMass(mass_structure)

		# self.F = np.zeros((20,38)) # selection matrix
		# for i in range(len(self.fixed_joint_indicies)):			
		# 	self.F[i,self.fixed_joint_indicies[i]] = 1

		# self.F_fixed = np.zeros((32,38)) # selection matrix
		# for i in range(len(self.all_fixed_joint_indicies)):			
		# 	self.F_fixed[i,self.all_fixed_joint_indicies[i]] = 1

	def get_world(self):
		return self.world_all_active

	def get_CoM(self):
		return self.robot_all_active.getCom()

	def set_q_2D(self,q):
		"""
		Parameters:
		--------------
		q: n*1 column vector; 2D numpy array
		"""
		q_to_be_set = [0.0]*self.N_of_joints_3D
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_to_be_set[i] = q[j,0]
		#print(q_to_be_set,len(q_to_be_set))
		self.robot_all_active.setConfig(q_to_be_set)

		return np.array(q_to_be_set)[np.newaxis].T

	def set_q_dot_2D(self,q_dot):
		# assert len(q_dot) == 15 , "wrong length for q"
		q_dot_to_be_set = [0.0]*self.N_of_joints_3D
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_dot_to_be_set[i] = q_dot[j,0]
		#print(q_dot_to_be_set, len(q_dot_to_be_set))
		self.robot_all_active.setVelocity(q_dot_to_be_set)

		return np.array(q_dot_to_be_set)[np.newaxis].T

	def set_q_2D_(self,q):
		"""
		Parameters:
		--------------
		q: 1D numpy array
		"""
		q_to_be_set = [0.0]*self.N_of_joints_3D
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_to_be_set[i] = float(q[j])
		#print(q_to_be_set,len(q_to_be_set))
		self.robot_all_active.setConfig(q_to_be_set)

		return np.array(q_to_be_set)[np.newaxis].T

	def set_q_dot_2D_(self,q_dot):
		"""
		Parameters:
		--------------
		q: 1D numpy array
		"""
		#assert len(q_dot) == 18 , "wrong length for q"
		q_dot_to_be_set = [0.0]*self.N_of_joints_3D
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_dot_to_be_set[i] = float(q_dot[j])
		#print(q_dot_to_be_set, len(q_dot_to_be_set))
		self.robot_all_active.setVelocity(q_dot_to_be_set)

		return np.array(q_dot_to_be_set)[np.newaxis].T

	def get_mass_matrix(self):
		m_3D = self.robot_all_active.getMassMatrix()
		m = []
		for i in self.joint_indices_3D:
			row = []
			for j in self.joint_indices_3D:
				row.append(m_3D[i][j]) 
			m.append(row)
		return m

	def get_ankle_positions(self,full = False):
		"""return the 2D/3D positions of the 4 ankle sole positions. If full == True, return the 3D position

		returns:
		------------
		a 4*3 list or 4*6 list
		"""
		positions = []
		if self.print_level == 1:
			print('current robot q when calculating ankle positions:',self.robot_all_active.getConfig())
		if full:
			for i in range(2):
				p = self.robot_all_active.link(13+i*8).getWorldPosition((0.15,0.0,0.0)) 
				direction = self.robot_all_active.link(13+i*8).getWorldDirection((-0.15,0.0,0.0))
				a = math.atan2(direction[0],direction[2])
				positions.append(p+direction)
		else:
			for i in range(2):
				p = self.robot_all_active.link(13+i*8).getWorldPosition((0.15,0.0,0.0)) 
				direction = self.robot_all_active.link(13+i*8).getWorldDirection((-0.15,0.0,0.0))
				a = math.atan2(direction[0],direction[2])
				positions.append([p[0],p[2],a])

		return positions



	def get_Jacobians(self):
		local_pt = (0,0,0)
		J1 = np.array(self.robot_all_active.link(13).getJacobian(local_pt))
		J2 = np.array(self.robot_all_active.link(21).getJacobian(local_pt))
		J3 = np.array(self.robot_all_active.link(29).getJacobian(local_pt))
		J4 = np.array(self.robot_all_active.link(37).getJacobian(local_pt))
		J1 = J1[[3,5,1],:] #orientation jacobian is stacked upon position
		J2 = J2[[3,5,1],:]
		J3 = J3[[3,5,1],:]
		J4 = J4[[3,5,1],:]
		return J1[:,self.joint_indices_3D]


	def compute_CD(self,u,gravity = (0,0,-9.81)):
		""" Compute the dynamics of the 2D robot, given by matrices C and D.
		acceleration = C + D*contact_wrench 

		Parameters:
		------------
		u a  numpy array
		
		Returns:
		------------
		C,D: numpy array nx1 2D arrays...
		"""
		#debug

		# print('robot config',self.robot_all_active.getConfig())
		# print('robot velocity',self.robot_all_active.getVelocity())

		u = self.u_2D_to_3D(u) #u is a 1D numpy array
		B_inv = np.array(self.robot_all_active.getMassMatrixInv())
		#print('coriolis forces',self.robot_all_active.getCoriolisForces())
		I = np.eye(38)

		a_from_u = np.array(self.robot_all_active.accelFromTorques(u))
		K = np.subtract(I,np.dot(B_inv,np.dot(self.F.T,np.dot(np.linalg.inv(np.dot(self.F,np.dot(B_inv,self.F.T))),self.F))))
		G = np.array(self.robot_all_active.getGravityForces(gravity))
		a = np.dot(K,a_from_u.T)
		#add coriolis + centrifugal
		Co = self.robot_all_active.getCoriolisForces()
		C = a - K@B_inv@(G+Co)

		self._clean_vector(C)
		J1 = np.array(self.robot_all_active.link(13).getJacobian((0.075,0,0)))
		J2 = np.array(self.robot_all_active.link(21).getJacobian((0.075,0,0)))
		J3 = np.array(self.robot_all_active.link(29).getJacobian((0.075,0,0)))
		J4 = np.array(self.robot_all_active.link(37).getJacobian((0.075,0,0)))
		J1 = J1[[3,5,1],:] #orientation jacobian is stacked upon position
		J2 = J2[[3,5,1],:]
		J3 = J3[[3,5,1],:]
		J4 = J4[[3,5,1],:]
		J = np.vstack((J1,np.vstack((J2,np.vstack((J3,J4))))))
		D = np.dot(K,np.dot(B_inv,J.T))

		C = C[np.newaxis].T
		return C[self.joint_indices_3D,:],D[self.joint_indices_3D,:]

	def compute_CD_fixed(self,gravity = (0,0,-9.81)):

		B_inv = np.array(self.robot_all_active.getMassMatrixInv())
		I = np.eye(38)
		K = np.subtract(I,B_inv@self.F_fixed.T@np.linalg.inv(self.F_fixed@B_inv@self.F_fixed.T)@self.F_fixed)
		G = np.array(self.robot_all_active.getGravityForces(gravity))
		Co = self.robot_all_active.getCoriolisForces()
		C = np.multiply(K@B_inv@(G+Co),-1.0)
		self._clean_vector(C)
		J1 = np.array(self.robot_all_active.link(13).getJacobian((0.075,0,0)))
		J2 = np.array(self.robot_all_active.link(21).getJacobian((0.075,0,0)))
		J3 = np.array(self.robot_all_active.link(29).getJacobian((0.075,0,0)))
		J4 = np.array(self.robot_all_active.link(37).getJacobian((0.075,0,0)))
		J1 = J1[[3,5,1],:] #orientation jacobian is stacked upon position
		J2 = J2[[3,5,1],:]
		J3 = J3[[3,5,1],:]
		J4 = J4[[3,5,1],:]
		J = np.vstack((J1,np.vstack((J2,np.vstack((J3,J4))))))
		D = K@B_inv@J.T

		##debug, add the torques for fixed joints
		L = np.linalg.inv(self.F_fixed@B_inv@self.F_fixed.T)@self.F_fixed@B_inv
		L_prime = L@(G+Co)
		L_J = np.multiply(L@J.T,-1.0)
		C = C[np.newaxis].T
		return C[self.joint_indices_3D,:],D[self.joint_indices_3D,:],L_prime,L_J

	def compute_Jp(self, contact_list):
		Jp = np.zeros((12,38))
		for i in contact_list:
			Jp[i*3:i*3+3,:] = np.array(self.robot_all_active.link(13+i*8).getJacobian((0.15,0,0)))[[3,5,1],:]
		return Jp[:,self.joint_indices_3D]

	def compute_Jp_Partial(self):
		"""
		return:
		---------
		return the 8 by 15 jacobian of the robot. the z position and angle

		"""

		Jp = np.zeros((8,38))
		for i in [0,1,2,3]:
			Jp[i*2:(i+1)*2,:] = np.array(self.robot_all_active.link(13+i*8).getJacobian((0.15,0,0)))[[5,1],:]
		return Jp[:,self.joint_indices_3D]

	def compute_Jp_Partial2(self):
		"""
		return:
		---------
		return the 8 by 15 jacobian of the robot. the x,z position

		"""

		Jp = np.zeros((8,38))
		for i in [0,1,2,3]:
			Jp[i*2:(i+1)*2,:] = np.array(self.robot_all_active.link(13+i*8).getJacobian((0.15,0,0)))[[3,5],:]
		return Jp[:,self.joint_indices_3D]		

	def _clean_vector(self,a):
		for i in range(len(a)):
			if math.fabs(a[i]) < 0.000000001:
				a[i] = 0.0

	def q_3D_to_2D(self,q):
		q_2D = []
		for i in self.joint_indices_3D:
			q_2D.append(q[i,0])
		return q_2D

	def q_2D_to_3D_(self,q):
		"""
		Parameters:
		q is list/1D array

		Return:
		list
		"""
		q_3D = [0]*38
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_3D[i] = q[j]
		return q_3D

	def q_2D_to_3D(self,q):
		"""
		Parameters:
		q column vector

		Return:
		list
		"""
		q_3D = [0]*38
		for (i,j) in zip(self.joint_indices_3D,self.joint_indices_2D):
			q_3D[i] = q[j,0]
		return q_3D


	def u_2D_to_3D_column(self,u):
		"""
		Parameters:
		-----------------
		U is a n*1 column vector.
		"""
		u_3D = [0]*self.N_of_joints_3D
		for (i,j) in zip(self.revolute_joint_indices_3D,self.revolute_joint_indices_2D):
			u_3D[i] = u[j,0]
		return u_3D

	def u_2D_to_3D(self,u):
		"""
		Parameters:
		-----------------
		U is a numpy array
		"""
		u_3D = [0.0]*self.N_of_joints_3D
		for (i,j) in zip(self.revolute_joint_indices_3D,self.revolute_joint_indices_2D):
			u_3D[i] = u[j]
		return u_3D

	def do_IK(self,R,t,limb):
		"""
		limb: int, 0-3

		"""
		current_config = self.robot_all_active.getConfig()
		active_Dofs = self.active_joint_indices[limb*3:limb*3+3]
		link = self.robot_all_active.link(13+8*limb)
		goal = ik.objective(link,R=R,t = t)
		if ik.solve(goal,activeDofs = active_Dofs,tol = 5e-3):
			target_config = self.robot_all_active.getConfig()
			target_config_2D = []
			for index in self.joint_indices_3D:
				target_config_2D.append(target_config[index])
		else:
			self.robot_all_active.setConfig(current_config)
			print('IK solve failure: no IK solution found')
			return
		self.robot_all_active.setConfig(current_config)
		return target_config_2D










if __name__=="__main__":
	robot = robosimian()
	#robot.set_generalized_q([0]*15)
	#robot.compute_CD(np.zeros((12,1)))
