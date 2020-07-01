import math
import numpy as np
from copy import deepcopy
import configs
import matplotlib.pyplot as plt
from klampt.math import vectorops as vo
from scipy.sparse import coo_matrix
from robosimian_wrapper import robosimian
def initialize():
	from robosimian_GM_simulation import robosimianSimulator
	#global robot
	#these q's don't matter
	q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
		+[0.6+1.5708,0.0,-0.6])[np.newaxis].T
	q_dot_2D = np.array([0.0]*15)[np.newaxis].T

	global robot
	robot = robosimianSimulator(q = q_2D,q_dot= q_dot_2D,dt = 0.005,solver = 'cvxpy',print_level = 1, augmented = True)

def jac_dyn(x, u ,eps = 1e-4):
	global robot
	#calculate accleration and dynamics jacobian
	a,J_SA = robot.getDynJac(x,u)

	#a,C,D,wc = robot.getDyn(x,u)
	#print('accleration:',a)
	print('------------------------------------------')
	a = np.concatenate([x[15:30],np.ravel(a)])		

	#calculate jacobian with finite-difference
	eps = 1e-5
	J = np.zeros((30,30+12))
	for i in [0,1,2,3,4]:#range(30):
		FD_vector = np.zeros(30)
		FD_vector[i] = eps
		tmp_x = np.add(x,FD_vector)
		tmp_a,_,_,_= robot.getDyn(tmp_x,u)
		#print('accleration:',tmp_a)
		J[:,i] = np.multiply(np.subtract(np.concatenate([tmp_x[15:30],np.ravel(tmp_a)]),a),1.0/eps)
		print('-----------------------',i,'-------------------')
	print('FD')
	print(J[15:30,0:4])
	# print(J[15:30,4:8])
	print('SA')
	print(J_SA[15:30,0:4])
	# print(J_SA[15:30,4:8])

	print(J_SA[0,:])
	print(np.shape(J_SA))
	return

def jacobian(x,u):
	jac_dyn(x, u ,eps = 5e-4)
	jac_dyn(x, u ,eps = 1e-4)
	jac_dyn(x, u ,eps = 1e-5)


def compute(x,u):
	global robot
	#compute stuff
	#check how acceleration changes as the state changes
	dimension = 3

	x2_start = x[dimension]
	deltas = [1e-7,1e-6,5e-6,1e-5,5e-5,1e-4]
	#deltas = [1e-4]#,3.6e-6,3.7e-6]
	x_axis = []
	counter = 0
	for delta in deltas:
	    x[dimension] = x2_start + delta
	    a,_,_,_ = robot.getDyn(x,u)
	    if counter == 0:
	        As = a
	    else:
	        As = np.hstack((As,a))
	    x_axis.append(x[dimension] - x2_start)
	    counter = counter + 1
	np.save('temp_files/y_axis',As)
	np.save('temp_files/x_axis',x_axis)

def plot():
	As = np.load('temp_files/y_axis.npy')
	x_axis = np.load('temp_files/x_axis.npy')
	print(x_axis)
	plt.plot(np.log10(x_axis),As[0,:],np.log10(x_axis),As[1,:],np.log10(x_axis),As[2,:],np.log10(x_axis),As[3,:])
	plt.legend(['a1','a2','a3','a4'])
	plt.ylabel('acceleration')
	plt.xlabel('log(delta q4)')
	plt.title('Acceleration changes')
	plt.show()

class transportationCost():
	def __init__(self):
		self.nX = 30
		self.nU = 12
		self.nP = 0
		self.N = 181
		self.first_q_dot = 18
		self.NofJoints = 12
		self.first_u = 30
		self.Gnzz = self.N*(self.nX+self.nU)-self.N*self.first_q_dot +2 
		#NonLinearObj.__init__(self,nsol = self.N*(self.nX+self.nU+self.nP),nG = self.N*(self.nX+self.nU)-self.N*self.first_q_dot +2  )
		#setting 16
		self.scale = 100.0

		#TODO, need to modify the gradient for adding small_C
		self.small_C = 0.01
		#print(self.N*(self.nX+self.nU)-self.N*self.first_q_dot +2)

	def __callg__(self,x, F, G, row, col, rec, needg):
		effort_sum = 0.0
		for i in range(self.N):
			for j in range(self.NofJoints):
				#q_dot * u
				effort_sum = effort_sum + (x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
					x[i*(self.nX+self.nU+self.nP)+self.first_u+j])**2
				# if i*(self.nX+self.nU+self.nP)+self.first_q_dot+j == 7243:
				# 	print('debug:')
				# 	print(x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j],x[i*(self.nX+self.nU+self.nP)+self.first_u+j])
				# 	print((x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*x[i*(self.nX+self.nU+self.nP)+self.first_u+j])**2s)
		#print('from within the function:',effort_sum)

		F[:] = effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)/self.scale
		if needg:
			Gs = []
			nonzeros = [0]
			Gs.append(effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)**2)
			d = x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C
			for i in range(self.N-1):
				for j in range(self.NofJoints):
					Gs.append(2.0/d*x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j]**2)
					nonzeros.append(int(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j))
				for j in range(self.NofJoints):
					Gs.append(2.0/d*(x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]**2)*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j])
					nonzeros.append(int(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j+self.NofJoints))
			Gs.append(-effort_sum/(x[(self.N-1)*(self.nX+self.nU+self.nP)]-x[0]+self.small_C)**2)

			nonzeros.append(int((self.N-1)*(self.nX+self.nU+self.nP)))
			for i in [self.N-1]:
				for j in range(self.NofJoints):
					Gs.append(2.0/d*x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j]**2)
					nonzeros.append(int(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j))
				for j in range(self.NofJoints):
					Gs.append(2.0/d*(x[i*(self.nX+self.nU+self.nP)+self.first_q_dot+j]**2)*\
						x[i*(self.nX+self.nU+self.nP)+self.first_u+j])
					nonzeros.append(int(i*(self.nX+self.nU+self.nP)+self.first_q_dot+j+self.NofJoints))
			G[:] = vo.div(Gs,self.scale)
			if rec:
				row[:] = [0]*self.Gnzz
				col[:] = nonzeros

class anklePoseConstr():
	def __init__(self):

		lb = np.array([-0.2,-1.0,-0.2,-1.0,-0.2,-1.0,-0.2,-1.0])
		ub = np.array([1.0]*8)
		self.robot = robosimian()
		#NonLinearPointConstr.__init__(self,index = 0, nc = 8, nx = 30, nu = 12, np = 0 ,lb = lb, ub = ub, nG = 40)

	def __callg__(self,x, F, G, row, col, rec, needg):
		#first column of G is w.r.t. to time
		self.robot.set_q_2D_(x[0:15])
		self.robot.set_q_dot_2D_(x[15:30])
		p = self.robot.get_ankle_positions()
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

if __name__=="__main__":
	##These are for debugging the dynamcis gradients

	#x_new = vo.add(configs.q_staggered_limbs,[0.1,0.0,0.05,0.02,-0.03,-0.3,-0.1,0.15,-0.2,-0.3,0.25,-0.4,-0.07,0.2,-0.2])
	#x0 = np.array(x_new+[0.0]*15) #0.936 -- -0.08 ankle depth
	#x0 = np.array(configs.q_staggered_limbs+[0.0]*15) #0.936 -- -0.08 ankle depth
	#x0[1] = 0.915
	#x0 = x0 + np.random.rand(30)*0.1
	#x0[1] = 1.0
	#u0 = np.array(configs.u_augmented_mosek) + np.array(np.random.rand(12))
	#u0 = np.array(np.random.rand(12))
	#u0 = np.zeros(12)
	#u0 = np.array([6.08309021,0.81523653, 2.53641154 ,5.83534863 ,0.72158568, 2.59685143,\
	#	5.50487329, 0.54710471,2.57836468, 5.75260704, 0.64075017, 2.51792186])
	# x0 = np.array(configs.q_symmetric+[0.0]*15)
	# u0 = np.array([6.08309021,0.81523653, 2.53641154 ,-5.50487329, -0.54710471,-2.57836468,\
	#  	5.50487329, 0.54710471,2.57836468, -6.08309021,-0.81523653, -2.53641154])
	#u0 = np.zeros(12)


	#initialize()
	# compute(x0,u0)
	# plot()
	#jac_dyn(x0,u0)


	######This is for debugging the other constraints and objective functions
	#load the initial guess for test 16:
	# traj = np.hstack((np.load('results/PID_trajectory/2/q_init_guess.npy'),np.load('results/PID_trajectory/2/q_dot_init_guess.npy')))
	# u = np.load('results/PID_trajectory/2/u_init_guess.npy')
	# N = 181
	# nnz = 4346
	# nG_full = N*42
	# print('The size of the G should be ',N*42)
	# x = np.array([])
	# for i in range(181):
	# 	x = np.concatenate((x,traj[i,:],u[i,:]))
	# print('shape of x is',np.shape(x))
	# cost = transportationCost()
	# #cost = anklePoseConstr()
	# F = np.zeros(1)
	# G = np.zeros(nnz)
	# row = np.zeros(nnz,dtype = 'int')
	# col = np.zeros(nnz,dtype = 'int')
	# cost.__callg__(x, F, G, row, col, rec = True, needg = True)

	# G_auto = coo_matrix((G,(row,col)),shape = (1,nG_full))
	# G_auto_full = G_auto.toarray().flatten()

	
	# eps = 1e-7
	# G_FD_full = []
	# for i in range(nG_full):
	# 	FD_vector = np.zeros(nG_full)
	# 	FD_vector[i] = eps
	# 	new_F = np.zeros(1)
	# 	cost.__callg__(x+FD_vector, new_F, G, row, col, rec = False, needg = False)	 
	# 	G_FD_full.append((new_F[0]-F[0])/eps)
	# G_FD_full = np.array(G_FD_full)
	# diff = G_FD_full-G_auto_full

	# relative_diff = []
	# for i in range(nG_full):
	# 	if math.fabs(G_auto_full[i]) < 1e-10:
	# 		if math.fabs(G_FD_full[i]) > 1e-6:
	# 			print("problem:",G_FD_full[i])
	# 		relative_diff.append(0.0)
	# 	else:
	# 		rd = math.fabs(diff[i]/G_auto_full[i])
	# 		relative_diff.append(rd)

	# 		if rd > 0.01:
	# 			if math.fabs(G_auto_full[i]) > 0.01:
	# 				print('-----'+str(i)+'------')
	# 				print('relative difference is ',rd)
	# 				print(G_FD_full[i],G_auto_full[i])

	# print('maximum relative differernce:',np.max(relative_diff))
	# ind  = np.argmax(relative_diff)
	# print('index is',ind)
	# print(G_FD_full[ind],G_auto_full[ind])

	# print('maximum differernce:',np.max(diff))
	# print('The gradient from FD here is:',G_FD_full[np.argmax(diff)])
	# print('The gradient from user G here is:',G_auto_full[np.argmax(diff)])

	##### this is to look at the anklePoseConstr
	# traj = np.hstack((np.load('results/PID_trajectory/2/q_init_guess.npy'),np.load('results/PID_trajectory/2/q_dot_init_guess.npy')))
	# u = np.load('results/PID_trajectory/2/u_init_guess.npy')
	# cost = anklePoseConstr()
	# nnz = 40
	# nG_full = 8*43
	# max_diffs = []
	# for iter1 in range(181):
	# 	x = np.array([])
	# 	x = np.concatenate((x,traj[iter1,:],u[iter1,:]))
	# 	F = np.zeros(8)
	# 	G = np.zeros(nnz)
	# 	row = np.zeros(nnz,dtype = 'int')
	# 	col = np.zeros(nnz,dtype = 'int')
	# 	cost.__callg__(x, F, G, row, col, rec = True, needg = True)

	# 	G_auto = coo_matrix((G,(row,col)),shape = (8,43))
	# 	G_auto_full = G_auto.toarray()[:,1:43].flatten()

	# 	eps = 1e-6
	# 	diffs = []
	# 	for i in range(42):
	# 		FD_vector = np.zeros(42)
	# 		FD_vector[i] = eps
	# 		new_F = np.zeros(8)
	# 		cost.__callg__(x+FD_vector, new_F, G, row, col, rec = False, needg = False)	 
	# 		diff = (new_F[0]-F[0])
	# 		diffs.append(max(diff.min(), diff.max(), key=abs))
	# 	max_diffs.append((np.max(np.array(diffs))))	
	# print(np.max(np.array(max_diffs)))


	# knitro_obj = np.load('debug/knitro_obj.npy')
	# knitro_con = np.load('debug/knitro_con.npy')

	# solverlib_obj = np.load('debug/solverlib_obj.npy')
	# solverlib_con = np.load('debug/solverlib_con.npy')

	# print('knitro obj:',knitro_obj)
	# #print('solverlib obj:',solverlib_obj)

	# x = np.load('debug/solution_x2.npy')
	# u = np.load('debug/solution_u2.npy')

	# print('x0[0],xf[0]',x[0,0],x[-1,0])
	# #The first of the constraints is the objective value
	# #The last of the constraints is something added by trajOptLib
	# dyn_constr = knitro_con[1:5401]
	# print('Max Dynamics Constr:',np.sort(np.array(dyn_constr)))
	# print('Ankle Pose Constr:',knitro_con[5401:5401+181*8])
	# print('Enough Translation Constr:',knitro_con[5401+181*8:5401+181*8+1])

	raise Exception('xxx')
	print('flag')