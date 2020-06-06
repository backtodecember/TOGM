#Note that pebbles have not been trained yet for polyeven with eta = 4
import numpy as np
import math
from copy import deepcopy
from klampt.math import so2
from klampt.math import vectorops as vo 
import scipy.io as sio
from scipy.spatial import ConvexHull
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
class granularMedia:
	def __init__(self,material = "sand",print_level = 0,augmented = False,extrapolation = False):
		self.extrapolation = extrapolation #linearly scale the wrench torque
		#self.extrapolation_factor = 2.5
		self.print_level = print_level
		self.augmented = augmented
		if augmented:
			mat_contents = sio.loadmat('data/data&weightsSand.mat')
			D = mat_contents['D']
			self.cT = D['cT'][0][0][0][0]
			self.Nvel = D['Nvel'][0][0][0][0]

			self.material_range = [-0.07,0.9]
			#self.W = np.load('data/sandPolyevenWAugmented.npy') #this one does not have the zeros
			if self.extrapolation:
				self.W = np.load('data/sandPolyevenWAugmented3.npy')
				self.theta = np.load('data/sandPolyevenThetaAugmented3.npy')
			else:
				self.W = np.load('data/sandPolyevenWAugmented2.npy')
				self.theta = np.load('data/sandPolyevenThetaAugmented2.npy')
			# if self.extrapolation:
			# 	self.W = self.W*self.extrapolation_factor
			self.func = 4 #polyeven
			self.eta = 2

			#self.theta = np.load('data/sandPolyevenThetaAugmented.npy')
			#self.Ntheta = 33
			#self.scale = 17.0
			self.scale = 5.0
			
			self.Ntheta = 168
		else:
			if material == "sand":
				mat_contents = sio.loadmat('data/data&weightsSand.mat')
				self.material_range = [-0.07,0.9]
			elif material == "pebbles":
				mat_contents = sio.loadmat('data/data&weightsPebbles.mat')
				self.material_range = [-0.045,0.9]
			D = mat_contents['D']
			self.data = D['data'][0][0]
			if material == "sand":
				#self.W = D['W'][0][0]
				#self.eta = D['eta'][0][0][0][0]
				#self.func = D['func'][0][0][0][0] - 1 #matlab is 1-based...
				#polyeven extrapolates better
				self.W = np.load('data/sandPolyevenW.npy')
				self.func = 4 #polyeven
				self.eta = 4
				self.scale =17.0
			else:
				#The pebbles seems too rigid
				self.W = np.load('pebblesLinearW.npy')
			self.theta = D['theta'][0][0]
			self.Nvel = D['Nvel'][0][0][0][0]
			self.Ntheta = D['Ntheta'][0][0][0][0]
			
			#print("func is",self.func)
			self.cT = D['cT'][0][0][0][0]
			self.estimated_upper_bound = 50 #max number of triangle meshes in convex hull
		#compute the things SA needs...




	def feasibleWrenchSpace(self,contact,ankle_length,compute_flag):
		"""Contact is angle, depth, which half, which leg, contact wrench rotation"""
		"""Matlab package has interpolation, do not do that yet here"""

		formulation = 'V'
		if compute_flag:
			if formulation == 'H':
				theta_new = contact[0:2]
				wrenches = np.zeros((self.Nvel,3))
				#slope_angle = contact[4]
				for iteration in range(self.Nvel):
					weights = self.W[iteration*self.Ntheta:iteration*self.Ntheta+self.Ntheta]
					val,grad = self._RBF(theta_new,self.theta[iteration2],self.eta,self.func)
					p = [0,0,0] 
					for iteration2 in range(self.Ntheta):
						p = vo.add(p,vo.mul(weights[iteration2],val))
					#p[2] = -p[2]  #the torque defined when robosimian is collecting data is y, into the page.. no longer needs to flip here
					unit_direction =  [-math.sin(theta_new[1]),0,-math.cos(theta_new[1])]
					tmp = vo.cross(vo.mul(unit_direction,ankle_length/2.0-self.cT),[p[0],0,p[1]])
					p[2] = p[2] + tmp[2]
					wrenches[iteration] = p

					#TODO: for a slope, we would need to do this..
					#wrenches[iteration][0:2] = so2.apply(wrenches[iteration][0:2],slope_angle)

					#deal with the mirror configuration, flip x force and torque. z force can stay the same..
					if contact[2] < 0:
						wrenches[iteration,0] = -wrenches[iteration,0]
						wrenches[iteration,2] = -wrenches[iteration,2]

				# average_wrench = self._average(wrenches)

				# plot_flag = True
				# if plot_flag:		
				# 	fig = plt.figure()
				# 	ax = plt.axes(projection = '3d')
				# 	ax.scatter3D(wrenches[:,0],wrenches[:,1],wrenches[:,2],c = 'g')
				# 	ax.scatter3D(average_wrench[0],average_wrench[1],average_wrench[2],c='r')
				# 	ax.set_xlabel('Fx (N)')
				# 	ax.set_ylabel('Fz (N)')
				# 	ax.set_zlabel('Torque (Nm)')

				K = ConvexHull(wrenches).simplices

				# #Plot to debug

				# if plot_flag:
				# 	for tri in K:
				# 		x = [wrenches[tri[0],0],wrenches[tri[1],0],wrenches[tri[2],0],wrenches[tri[0],0]]
				# 		y = [wrenches[tri[0],1],wrenches[tri[1],1],wrenches[tri[2],1],wrenches[tri[0],1]]
				# 		z = [wrenches[tri[0],2],wrenches[tri[1],2],wrenches[tri[2],2],wrenches[tri[0],2]]
				# 		ax.plot3D(x,y,z,'red')
				# 	plt.show()
				########################


				
				(m,n) = np.shape(K)
				if m > self.estimated_upper_bound:
					print('WS estimated_upper_bound exceeded..')
					return 
				A = np.zeros((self.estimated_upper_bound,n))
				b = np.zeros((self.estimated_upper_bound,1))

				for j in range(m):
					line1 = wrenches[K[j,1],:] - wrenches[K[j,0],:]
					line2 = wrenches[K[j,2],:] - wrenches[K[j,1],:]
					normal = vo.cross(line1,line2)
					normal = vo.div(normal,vo.norm(normal))
					if vo.dot(vo.sub(average_wrench,wrenches[K[j,1],:]),normal) >= 0:
						D = -vo.dot(normal,wrenches[K[j,1],:])
						A[j,:] = vo.mul(normal,-1.0)
						b[j] = D
						

					else:
						D = vo.dot(normal,wrenches[K[j,1],:])
						A[j,:] = normal
						b[j] = D
				return A,b
			elif formulation == 'V':
				theta_new = contact[0:2]
				wrenches = np.zeros((self.Nvel,3))
				#SA
				Q4s = np.zeros((self.Nvel,6))
				#slope_angle = contact[4]
				for iteration in range(self.Nvel):
					weights = self.W[iteration*self.Ntheta:iteration*self.Ntheta+self.Ntheta]
					p = [0,0,0] 
					Q4 = np.zeros((3,2))
					# debug
					# if self.print_level == 1:
					# 	if iteration == 9:
					# 		print('---- in utilities-----')
					for iteration2 in range(self.Ntheta):
						val,grad = self._RBF(theta_new,self.theta[iteration2],self.eta,self.func)
						p = vo.add(p,vo.mul(weights[iteration2],val))
						#SA
						w = np.array(weights[iteration2])[np.newaxis].T
						Q4 = Q4 + np.dot(w,np.array(grad)[np.newaxis])

					# if self.print_level == 1:
					# 	if iteration == 9:
					# 		print('----------------------')
					# 		print('Q4',Q4)
					#p[2] = -p[2]  #the torque defined when robosimian is collecting data is y, into the page.. no longer needs to flip here
					unit_direction =  [-math.sin(theta_new[1]),0,-math.cos(theta_new[1])]
					tmp = vo.cross(vo.mul(unit_direction,ankle_length/2.0-self.cT),[p[0],0,p[1]])
					p[2] = p[2] + tmp[2]
					wrenches[iteration] = p

					#TODO: for a slope, we would need to do this..
					#wrenches[iteration][0:2] = so2.apply(wrenches[iteration][0:2],slope_angle)

					#deal with the mirror configuration, flip x force and torque. z force can stay the same..
					if not self.augmented:
						if contact[2] < 0:
							wrenches[iteration,0] = -wrenches[iteration,0]
							wrenches[iteration,2] = -wrenches[iteration,2]

					Q4s[iteration,:] = self._vectorize(Q4)


				# plot_flag = True
				# average_wrench = self._average(wrenches)
				# print('The wrenches are:',wrenches)
				# if plot_flag:		
				# 	fig = plt.figure()
				# 	ax = plt.axes(projection = '3d')
				# 	ax.scatter3D(wrenches[:,0],wrenches[:,1],wrenches[:,2],c = 'g')


				# 	#debug
				# 	# wc2 = [6.74352075e+00,1.98369739e+02,7.73679077e-02]

				# 	# wc = [-35.24959736,243.80257131,-10.48360171]
				# 	# ax.scatter3D(wc[0],wc[1],wc[2],c='r')
				# 	# ax.scatter3D(wc2[0],wc2[1],wc2[2],c='b')
				# 	#ax.scatter3D(average_wrench[0],average_wrench[1],average_wrench[2],c='r')
				# 	ax.set_xlabel('Fx (N)')
				# 	ax.set_ylabel('Fz (N)')
				# 	ax.set_zlabel('Torque (Nm)')

				# K = ConvexHull(wrenches).simplices

				# #Plot to debug

				# if plot_flag:
				# 	for tri in K:
				# 		x = [wrenches[tri[0],0],wrenches[tri[1],0],wrenches[tri[2],0],wrenches[tri[0],0]]
				# 		y = [wrenches[tri[0],1],wrenches[tri[1],1],wrenches[tri[2],1],wrenches[tri[0],1]]
				# 		z = [wrenches[tri[0],2],wrenches[tri[1],2],wrenches[tri[2],2],wrenches[tri[0],2]]
				# 		ax.plot3D(x,y,z,'red')
				# 	plt.show()

				return wrenches.T,Q4s

		else:
			return 0,0

	def _average(self,W):
		counter = 0
		total = [0,0,0]
		for w in W:
			total = vo.add(total,w)
			counter = counter + 1
		return vo.div(total,counter)

	def _RBF(self,theta1,theta2,eta,func):
		#perform scaling here:
		scale = self.scale #17.0
		_theta1 = [theta1[0]*scale,theta1[1]]
		_theta2 = [theta2[0]*scale,theta2[1]]
		##func is 0 ....
		r = vo.norm(vo.sub(_theta1,_theta2))
		#out = [r,math.exp(-(eta*r)**2),math.sqrt(1+(eta*r)**2)]

		if func == 4: #polyeven
			if self.augmented:
				dphidr = math.log(r**r)+r*(math.log(r)+1) #eta = 2
			else:
				dphidr = 3*r**2*math.log(r**r)+r**3*(math.log(r)+1) #eta = 4

			#the bug is here.. need to scale....

			drdtheta_new = vo.mul(vo.sub(_theta1,_theta2),1.0/r)
			drdtheta_new[0] = drdtheta_new[0]*scale
			return r**(eta-1)*math.log(r**r),vo.mul(drdtheta_new,dphidr) #RBF and its gradient...
		else:
			return r , []
	def _vectorize(self,Q):
		"""
		Parameter:
		Q: numpy matrix

		Return:
		v: 1D numpy array
		"""

		(m,n) = np.shape(Q)
		v = []
		for j in range(n):
			for i in range(m):
				v.append(Q[i,j]) #column major
		return np.array(v)
if __name__=="__main__":
	# #check the validity of the robto kinematics..
	# mode = "fixed"
	# robot = robosimian(mode = mode)
	# dt = 0.01
	# q_generalized_1 = [0]*15
	# q_generalized_1[4] = 1.0
	# q_dot_generalized = [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
	# q_generalized_2 = vo.add(q_generalized_1,vo.mul(q_dot_generalized,dt))

	# robot.set_generalized_q(q_generalized_2)
	# robot.set_generalized_q_dot(q_dot_generalized)

	# #ground truth..
	# q_full_k = robot.q_full
	# q_dot_full_k = robot.q_dot_full

	# robot.set_generalized_q(q_generalized_1)
	# robot.set_generalized_q_dot([0]*15)
	# robot.set_all_q(q_full_k,q_dot_full_k,dt)
	# q_full_k_prime = robot.q_full
	# q_dot_full_k_prime = robot.q_dot_full
	# #print(robot.q_dot_generalized)
	# #print(q_full_k_prime)
	# #print(q_dot_full_k)
	# print(vo.norm(vo.sub(q_full_k_prime,q_full_k)))

	##debug terrain
	terrain = granularMedia()
	#print(np.shape(terrain.W[:,0:2]))
	#print(terrain.func)
	#for i in [-0.001,-0.002,-0.01,-0.02,-0.03,]:
	#	terrain.feasibleWrenchSpace([i,0.2,1,1,0],0.15)[0]
		#print(terrain.feasibleWrenchSpace([i,0.3,1,1,0],0.15)[0])
		#print(terrain.feasibleWrenchSpace([i,0.3,1,1,0],0.15)[1])