#klampt wrappper
from klampt.math import vectorops as vo
import numpy as np
import math
import pydrake as pd
#from pydrake.multibody.rigid_body_tree import RigidBodyTree
#from pydrake.autodiffutils import AutoDiffXd as AD

class robosimian_drake:
	def __init__(self,dt = 0.01):
		path = "data/robosimian_caesar_new_pinocchio.urdf"
		self.tree = pd.multibody.RigidBodyTree(path)
	def get_world(self):
		return self.world_all_active

	def set_q_2D(self,q):
		"""
		Parameters:
		--------------
		q: n*1 column vector; 2D numpy array
		"""
		# assert len(q) == 15 , "wrong length for q"
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
		# assert len(q) == 15 , "wrong length for q"
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
		# assert len(q_dot) == 15 , "wrong length for q"
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
		u = self.u_2D_to_3D(u) #u is a 1D numpy array
		B_inv = np.array(self.robot_all_active.getMassMatrixInv())
		I = np.eye(38)
		a_from_u = np.array(self.robot_all_active.accelFromTorques(u))
		K = np.subtract(I,np.dot(B_inv,np.dot(self.F.T,np.dot(np.linalg.inv(np.dot(self.F,np.dot(B_inv,self.F.T))),self.F))))
		G = np.array(self.robot_all_active.getGravityForces(gravity))
		a = np.dot(K,a_from_u.T)
		C = np.subtract(a,np.dot(K,np.dot(B_inv,G)))
		# print(self.robot_all_active.getConfig())
		# print(self.robot_all_active.getVelocity())
		#print('a',C)
		#print('C',C)

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


if __name__=="__main__":
	robot = robosimian_drake()
	#robot.set_generalized_q([0]*15)
	#robot.compute_CD(np.zeros((12,1)))
