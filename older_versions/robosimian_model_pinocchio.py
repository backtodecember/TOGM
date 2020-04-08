#A helper library that provides the kinematics and dynamics of a 2D robosimian model, based on Klampt.
from klampt import WorldModel
from klampt.math import vectorops as vo
import numpy as np
import math
import pinocchio
class PINOCCHIO_robosimian:
	def __init__(self,dt = 0.01):
		self.urdf_filename = 'data/robosimian_caesar_new_pinnochio.urdf'
		self.model = pinocchio.buildModelFromUrdf(urdf_filename)
		
		self.q = np.array([0.0]*15)
		self.v = np.array([0.0]*15)
		# ####link 7-13 RF; 15-21 RR; 23-29 LR; 31-37 LF
		self.dt = dt
		self.

	def set_q(self,a):
		"""
		Parameters:
		--------------
		q: n*1 column vector; 2D numpy array
		"""
		# assert len(q) == 15 , "wrong length for q"
		for i in range(15):
			self.q[i] = a[i,0]
		return 

	def set_q_dot_2D(self,q_dot):
		# assert len(q_dot) == 15 , "wrong length for q"
		for i in range(15):
			self.v[i] = q_dot[i,0]
		return 

	def set_q_(self,a):
		"""
		Parameters:
		--------------
		q: 1D array
		"""
		# assert len(q) == 15 , "wrong length for q"
		for i in range(15):
			self.q[i] = a[i]
		return 

	def set_q_dot_2D_(self,q_dot):
		# assert len(q_dot) == 15 , "wrong length for q"
		for i in range(15):
			self.v[i] = q_dot[i]
		return 

	def u_to_full_matrix(self,u):
		"""
		Parameters:
		---------------
		u:numpy 1D array, 12 elements

		Return:
		----------------
		numpy column vector, 19 elements
		"""
		u_full = 

	def compute_acceleration(self):



		return self.robot_all_active.getMassMatrix()

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
		print('flag')

		u = self.u_to_full_matrix(u) 
		B_inv = np.array(self.robot_all_active.getMassMatrixInv())
		I = np.eye(38)
		a_from_u = np.array(self.robot_all_active.accelFromTorques(u))
		K = np.subtract(I,B_inv@self.F.T@np.linalg.inv(self.F@B_inv@self.F.T)@self.F)
		G = np.array(self.robot_all_active.getGravityForces(gravity))
		a = K@a_from_u.T

		C = np.subtract(a,K@B_inv@G)
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
		return C[np.newaxis].T,D

	def compute_CD_g(self,u,q_dot_3D,gravity = (0,0,-9.81)):
		""" Compute the dynamics of the 2D robot, given by matrices C and D.
		acceleration = C + D*contact_wrench.
		Compared to compute_CD(), this function returns extra information 

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
		K = np.subtract(I,B_inv@self.F.T@np.linalg.inv(self.F@B_inv@self.F.T)@self.F)
		G = np.array(self.robot_all_active.getGravityForces(gravity))
		a = K@a_from_u.T
		C = np.subtract(a,K@B_inv@G)
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
		D = np.multiply(K@B_inv@J.T,self.dt)

		return np.add(np.multiply(C[np.newaxis].T,self.dt),q_dot_3D) ,D, dQ1dx

	def compute_CD_fixed(self,gravity = (0,0,-9.81)):

		B_inv = np.array(self.robot_all_active.getMassMatrixInv())
		I = np.eye(38)
		K = np.subtract(I,B_inv@self.F_fixed.T@np.linalg.inv(self.F_fixed@B_inv@self.F_fixed.T)@self.F_fixed)
		G = np.array(self.robot_all_active.getGravityForces(gravity))
		C = np.multiply(K@B_inv@G,-1.0)
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
		L_prime = L@G
		L_J = np.multiply(L@J.T,-1.0)

		return C[np.newaxis].T,D,L_prime,L_J

	def compute_Jp(self, contact_list):
		Jp = np.zeros((12,38))
		for i in contact_list:
			Jp[i*3:i*3+3,:] = np.array(self.robot_all_active.link(13+i*8).getJacobian((0.15,0,0)))[[3,5,1],:]
		return Jp

	def compute_Jp_2(self, contact_list):
		"""
		Return:
		Both 3D and 2D Jp
		"""
		Jp = np.zeros((12,38))
		for i in contact_list:
			Jp[i*3:i*3+3,:] = np.array(self.robot_all_active.link(13+i*8).getJacobian((0.15,0,0)))[[3,5,1],:]
		Jp_2D = Jp[:,self.joint_indices_3D]
		return Jp, Jp_2D

	def _clean_vector(self,a):
		for i in range(len(a)):
			if math.fabs(a[i]) < 0.000000001:
				a[i] = 0.0



if __name__=="__main__":
	robot = robosimian()
	#robot.set_generalized_q([0]*15)
	robot.compute_CD(np.zeros((12,1)))
