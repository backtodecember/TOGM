"""
This file includes the class of methods for visualizing and analyzing the results from 
a trajectory optimizer by only using the optimize x and u
"""
import numpy as np
from robosimian_GM_simulation import robosimianSimulator
import matplotlib.pyplot as plt
import configs
from copy import deepcopy
from klampt import vis
from klampt.model import trajectory
from klampt.math import vectorops as vo
import time
import matplotlib.pyplot as plt
import math
class analyzer:
	def __init__(self,case = '11-2',dt = 0.005,method = "Euler",x_data = [],u_data = []):
		self.dt = dt
		q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
		q_dot0 = np.zeros((15,1))
		self.robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = 0.05, solver = 'cvxpy', augmented = True, extrapolation= True)
		self.case = case

		if x_data != []:
			self.x = x_data
			self.u = u_data
			print('User-input got executed')
		else:
			self.u = np.load('results/'+case+'/solution_u.npy')
			self.x = np.load('results/'+case+'/solution_x.npy')
		(self.N,_) = np.shape(self.x)
		
		self.method = method
		self.iterations = np.arange(self.N)
		self.times = self.iterations*self.dt
		#slow down 20x

		#settings for animating the trajectory
		self.vis_dt = 0.005*10
		self.force_scale = 0.001 #200N would be 0.2m

		self.world = self.robot.getWorld()
		vis.add("world",self.world)

	def calculation(self):
		self.ankle_position_list = []
		self.force_list = []
		self.violations = []
		for i in range(self.N):
			self.robot.reset(self.x[i,0:15],self.x[i,15:30])
			ankle_positions = self.robot.robot.get_ankle_positions(full = True)
			force,a = self.robot.simulateOnce(self.u[i],continuous_simulation = False, SA = False, fixed = False)
			a = a.ravel()
			self.force_list.append(force)
			self.ankle_position_list.append(ankle_positions)
		
			if self.method == 'Euler':
				if i < self.N - 1:
					p_error = self.x[i+1,0:15]-self.x[i,0:15]-self.dt*self.x[i,15:30]
					v_error = self.x[i+1,15:30]-self.x[i,15:30]-self.dt*a
					self.violations.append(np.linalg.norm(np.concatenate((p_error,v_error))))
			elif self.method == 'BackEuler':
				if i > 0:
					p_error = self.x[i,0:15]-self.x[i-1,0:15]-self.dt*self.x[i,15:30]
					v_error = self.x[i,15:30]-self.x[i-1,15:30]-self.dt*a

					self.violations.append(np.linalg.norm(np.concatenate((p_error,v_error))))
			print('progress:',i/self.N)
		return 

	def animate(self):
		current_time = 0.0
		counter = 0
		self.robot.reset(self.x[counter,0:15],self.x[counter,15:30])
		vis.addText('time','time: '+str(0.0))
		vis.show()
		time.sleep(15)
		while vis.shown():
		#for i in range(self.N):
			vis.lock()
			if counter < self.N:
				self.robot.reset(self.x[counter,0:15],self.x[counter,15:30])
				ankle_positions = self.ankle_position_list[counter]
				force = self.force_list[counter]
				for i in range(4):
					force_vector = vo.mul(force[0+i*3:2+i*3],self.force_scale)
					limb_force = trajectory.Trajectory(times = [0,1],milestones = [vo.add(ankle_positions[i][0:3],[0,-0.1,0]),\
						vo.add(vo.add(ankle_positions[i][0:3],[force_vector[0],0,force_vector[1]]),[0,-0.1,0])])
					vis.add('force'+str(i+1),limb_force)
				current_time += self.dt
				vis.addText('time','time: '+str(current_time))
				counter += 1
			vis.unlock()
			time.sleep(self.vis_dt)
			

		vis.kill()
		return

	def dynConstrViolation(self,method = 'Euler'):
		print('Average Constraint Violations:',np.average(np.array(self.violations)))

		# plt.plot(self.times[0:self.N-1],self.violations)
		# plt.title('Dynamics Constraint Violation')
		# plt.grid()
		# plt.ylabel('Unscaled Constraint Violation 2-norm')
		# plt.xlabel('Time (s)')
		# plt.show()
		return

	def objective(self,x,u):
		"""
		This is problem specific
		"""
		scale = 100.0
		small_C = 0.01
		#below is the transportation cost
		effort_sum = 0.0
		#print(self.N)
		for i in range(self.N):
			for j in range(12):
				#q_dot * u
				effort_sum = effort_sum + (x[i,j+18]*u[i,j])**2
		obj = effort_sum/(x[self.N -1 ,0] - x[0,0] + small_C)/scale

		return obj*0.05**2							

	def otherConstr(self,type):
		"""
		Joint constraints/other
		Cyclic constraints.
		These are case specific
		"""
		if type == 'cyclic':
			vio = np.max(self.x[0,:]-self.x[-1,:])
		return vio

	def forcePlot(self):
		for limb in range(4):
			fig,axs = plt.subplots(3,1,sharex=True)
			axs[0].plot(self.times,self.force_list[limb*3+0,:])
			axs[0].set_title('x-force')
			axs[0].grid()
			axs[0].set(ylabel='N')
			axs[1].plot(self.times,self.force_list[limb*3+1,:])
			axs[1].set_title('z-force')
			axs[1].set(ylabel='N')
			axs[1].grid()
			axs[2].plot(self.times,self.force_list[limb*3+2,:])
			axs[2].set_title('torque') 
			axs[2].set(xlabel='Time (s)', ylabel='Nm')
			axs[2].grid()
			fig.suptitle('Limb ' + str(limb+1) +' Ground Reaction Force ')
			plt.show()

	def perIterationObj(self):
		#iterations = [0,5,10,15,20,24,28,31,35,40,44,46,49,53,56,60,62,65,67,69,70]
		#iterations = [0,5,10,15,20,25,35,40,45,50,55,60]#,65,70]#,75,80,85,90,95,100,105,110,115]

		iterations = []
		for i in range(17):
			iterations.append(int(i*5))
		objVals = []
		for iter in iterations:
			print('Iteration:',iter)
			if iter == 0:
				objVals.append(self.objective(np.load('results/PID_trajectory/2/x_init_guess.npy'),np.load('results/PID_trajectory/2/u_init_guess.npy')))
			else:
				objVals.append(self.objective(np.load('results/'+self.case+'/run3/solution_x'+str(iter)+'.npy'),np.load('results/'+self.case+'/run3/solution_u'+str(iter)+'.npy')))
				#objVals.append(self.objective(np.load('results/'+self.case+'/solution_x'+str(iter)+'.npy'),np.load('results/'+self.case+'/solution_u'+str(iter)+'.npy')))

		
		plt.plot(iterations,objVals)
		plt.title('Objective Values over Iterations')
		plt.grid()
		plt.ylabel('Objective')
		plt.xlabel('Iterations')
		plt.show()
		return

	def perIterGeneralConstrVio(self,type):
		iterations = []
		for i in range(41):
			iterations.append(int(i*5))
		if type == 'enough_translation':
			violations = []
			for iter in iterations:
				if iter == 0:
					x = np.load('results/PID_trajectory/3/x_init_guess.npy')
					diff = x[-1,0] - x[0,0] - 0.4
				else:
					x = np.load('results/17/solution_x'+str(iter)+'.npy')
					diff = x[-1,0] - x[0,0] - 0.4

				if diff < 0:
					violations.append(-diff)
				else:
					violations.append(0.0)
					plt.plot(iterations,violations)
			plt.title('Enough Translation Constraint Violations over Iterations')
			plt.grid()
			plt.ylabel('Violations')
			plt.xlabel('Iterations')
			plt.show()
		elif type == 'ankle_pose':
			from robosimian_wrapper import robosimian
			robot = robosimian()	
			lb = np.array([-0.2,-1.0,-0.2,-1.0,-0.2,-1.0,-0.2,-1.0])
			ub = np.array([1.0]*8)	
			violations = []					
			for iter in iterations:
				if iter == 0:
					Xs = np.load('results/PID_trajectory/3/x_init_guess.npy')
				else:
					Xs = np.load('results/17/solution_x'+str(iter)+'.npy')
					#Xs = np.load('results/16/run3/solution_x'+str(iter)+'.npy')
				violation = 0
				for x in Xs: 
					robot.set_q_2D_(x[0:15])
					robot.set_q_dot_2D_(x[15:30])
					p = robot.get_ankle_positions()
					p = np.array([p[0][1],p[0][2],p[1][1],p[1][2],p[2][1],p[2][2],p[3][1],p[3][2]])
					error = 0		
					for i in range(8):
						if p[i] > ub[i]:
							error += (p[i] - ub[i])**2
						if p[i] < lb[i]:
							error += (lb[i] - p[i])**2
				
					violation  += math.sqrt(error)
				violations.append(violation/self.N)

			plt.plot(iterations,violations)
			plt.title('Ankle Pose Constraint Violations over Iterations')
			plt.grid()
			plt.ylabel('Violations')
			plt.xlabel('Iterations')
			plt.show()


		print(violations)
			

			
	def perIterDynConstrVio(self):
		#iterations = [0,5,10,15,20,24,28,31,35,40,44,46,49,53,56,60,62,65,67,69,70]
		#iterations = [0,5,10,15,20,25,35,40,45,50,55,60]#,65,70]#,75,80,85,90,95,100,105,110,115]
		iterations = []
		for i in range(17):
			iterations.append(int(i*5))

		violations = []
		for iter in iterations:
			print('Iteration:',iter)
			if iter == 0:
				violations.append(self._dynConstrVio(np.load('results/PID_trajectory/2/x_init_guess.npy'),np.load('results/PID_trajectory/2/u_init_guess.npy')))
			else:
				violations.append(self._dynConstrVio(np.load('results/'+self.case+'/run3/solution_x'+str(iter)+'.npy'),np.load('results/'+self.case+'/run3/solution_u'+str(iter)+'.npy')))
				#violations.append(self._dynConstrVio(np.load('results/'+self.case+'/solution_x'+str(iter)+'.npy'),np.load('results/'+self.case+'/solution_u'+str(iter)+'.npy')))

		
		plt.plot(iterations,violations)
		plt.title('Dynamics Constraint Violations over Iterations')
		plt.grid()
		plt.ylabel('Violations')
		plt.xlabel('Iterations')
		plt.show()
		return

	def _dynConstrVio(self,x,u):
		violations = 0
		for i in range(self.N):
			print(i)
			self.robot.reset(x[i,0:15],x[i,15:30])
			force,a = self.robot.simulateOnce(u[i],continuous_simulation = False, SA = False, fixed = False)
			a = a.ravel()
			if self.method == 'Euler':
				if i < self.N - 1:
					p_error = self.x[i+1,0:15]-self.x[i,0:15]-self.dt*self.x[i,15:30]
					v_error = self.x[i+1,15:30]-self.x[i,15:30]-self.dt*a
					violations += np.linalg.norm(np.concatenate((p_error,v_error)))
			elif self.method == 'BackEuler':
				if i > 0:
					p_error = self.x[i,0:15]-self.x[i-1,0:15]-self.dt*self.x[i,15:30]
					v_error = self.x[i,15:30]-self.x[i-1,15:30]-self.dt*a

					violations += np.linalg.norm(np.concatenate((p_error,v_error)))
		return violations/self.N

#piece-wise constant trajectory
class pcTraj:
	def __init__(self,times,milestones):
		self.times = times
		self.milestones = milestones
		assert len(self.times)  == len(self.milestones)
		self.total_t = self.times[-1]
		self.total_N = len(self.times)

	def eval(self,t,c):
		assert t>= 0

		if c:
			if t > self.total_t:
				t = t%self.total_t
		
		else:
			assert t <= self.total_t
		
		for i in range(self.total_N):
			if t < self.times[i]:
				break
		
		if i < 1:
			return self.milestones[0]
		else:
			return self.milestones[i-1]
	

class PIDTracker:
	def __init__(self,case,visualization = True, traj_dt = 0.005, x_data = [],u_data = [], initial = True):
		"""
		The PID tracker tracks an optimized trajectory on a simulator using Euler integration with dt = 0.005
		"""
		self.visualization = visualization
		self.initial = initial
		self.dt = 0.005 #this is the default PID dt
		q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
		q_dot0 = np.zeros((15,1))
		self.robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = self.dt, solver = 'cvxpy', augmented = True, extrapolation= True)
		self.case = case
		if x_data != []:
			self.x = x_data
			self.u = u_data
		else:
			self.u = np.load('results/'+case+'/solution_u.npy')
			self.x = np.load('results/'+case+'/solution_x.npy')


		self.kp = np.array([1000.0,1000.0,1000.0]*4)
		self.ki = np.array([2.0]*12)
		self.kd = np.array([10.0,8.0,10.0]*4)

		(self.N,_) = np.shape(self.x)
		self.iterations = np.arange(self.N)
		self.times = self.iterations*traj_dt
		#slow down 20x
		self.ref_trajectory = trajectory.Trajectory(times = self.times ,milestones = self.x.tolist())
		#self.ref_torque = trajectory.Trajectory(times = self.times, milestones = self.u.tolist())  ## this actually does piece-wise linear interpolation
		self.ref_torque = pcTraj(times = self.times, milestones = self.u.tolist())  #this does piece-wise constant
		#settings for animating the trajectory
		#self.vis_dt = 0.005*40
		self.force_scale = 0.001 #200N would be 0.2m

		self.world = self.robot.getWorld()
		vis.add("world",self.world)

	def run(self):
		self.ankle_position_list = []
		self.force_list = []

		if self.visualization:
			vis.show()
			vis.addText('time','time: '+str(0.0))
			#vis.addGhost('ghost',robot = 0)
		#reset the robot
		self.robot.reset(self.x[0,0:15],self.x[0,15:30])
		#start simulation
		error = np.array([0.0]*12)
		last_error = np.array([0.0]*12)
		accumulated_error = np.array([0.0]*12)
		time.sleep(15.0)
		simulation_time = 0.0
		#start_time = time.time()

		#record these
		q_history = []
		q_dot_history = []
		u_history = []
		time_history = []

		if self.visualization:

			while vis.shown() and (simulation_time < self.times[-1]+0.0001):
				#loop_start_time = time.time()
				vis.lock()
				#simulation_time = time.time() - start_time
				current_q = self.robot.getConfig()[3:15] #1d array of 15 elements
				desired_q = np.array(self._targetQ(simulation_time))
				ghost_q = self.robot.getConfig()[0:3].tolist() + self._targetQ(simulation_time)
				full_ghost_q = self.robot.robot.q_2D_to_3D_(ghost_q)
				#print(full_ghost_q)
				vis.add('ghost',self.robot.robot.q_2D_to_3D_(ghost_q))
				vis.setColor('ghost',0,1,0,0.3)
				last_error = deepcopy(error)
				error = desired_q - current_q
				dError = (error - last_error)/self.dt
				accumulated_error += error
				u_raw = np.multiply(self.kp,error) + np.multiply(self.kd,dError) + np.multiply(self.ki,accumulated_error)
				tau = np.clip(u_raw,-200.0,200.0)
				#feed forward in one limb
				for i in range(4):
					tau[i*3+1] += tau[i*3+2] 
					tau[i*3] += tau[i*3+1]

				#add reference torque
				ref_tau = self.ref_torque.eval(simulation_time,True)
				#print('ref_torque',ref_tau)
				tau = tau + np.array(ref_tau)
				tau = np.clip(tau,-300.0,300.0) 

				#record 
				q_history.append(self.robot.getConfig().tolist())
				q_dot_history.append(self.robot.getVel().tolist())
				u_history.append(tau.tolist())
				time_history.append(simulation_time)

				simulation_time += self.dt
				vis.addText('time','time: '+str(simulation_time))

				force,a = self.robot.simulateOnce(tau,continuous_simulation = True)
				ankle_positions = self.robot.robot.get_ankle_positions(full = True)
				for i in range(4):
					force_vector = vo.mul(force[0+i*3:2+i*3],self.force_scale)
					limb_force = trajectory.Trajectory(times = [0,1],milestones = [vo.add(ankle_positions[i][0:3],[0,-0.1,0]),\
						vo.add(vo.add(ankle_positions[i][0:3],[force_vector[0],0,force_vector[1]]),[0,-0.1,0])])
					vis.add('force'+str(i+1),limb_force)
				
				# self.force_list.append(force)
				# self.ankle_position_list.append(ankle_positions)
				vis.unlock()
				time.sleep(0.001)
			while vis.shown():
				time.sleep(0.1)
			vis.kill()
		else:
			while (simulation_time < self.times[-1]+0.0001):
				
				current_q = self.robot.getConfig()[3:15] #1d array of 15 elements
				desired_q = np.array(self._targetQ(simulation_time))
				last_error = deepcopy(error)
				error = desired_q - current_q
				dError = (error - last_error)/self.dt
				accumulated_error += error
				u_raw = np.multiply(self.kp,error) + np.multiply(self.kd,dError) + np.multiply(self.ki,accumulated_error)
				tau = np.clip(u_raw,-200.0,200.0)
				#feed forward in one limb
				for i in range(4):
					tau[i*3+1] += tau[i*3+2] 
					tau[i*3] += tau[i*3+1]

				#add reference torque
				ref_tau = self.ref_torque.eval(simulation_time,True)
				#print('ref_torque',ref_tau)
				tau = tau + np.array(ref_tau)
				tau = np.clip(tau,-300.0,300.0) 

				#record 
				q_history.append(self.robot.getConfig().tolist())
				q_dot_history.append(self.robot.getVel().tolist())
				u_history.append(tau.tolist())
				time_history.append(simulation_time)

				simulation_time += self.dt
				vis.addText('time','time: '+str(simulation_time))

				force,a = self.robot.simulateOnce(tau,continuous_simulation = True)
				ankle_positions = self.robot.robot.get_ankle_positions(full = True)
				self.force_list.append(force)
				self.ankle_position_list.append(ankle_positions)
				print('Time',simulation_time)

		

		#save stuff
		self.q_history = np.array(q_history)
		self.q_dot_history = np.array(q_dot_history)
		self.u_history = np.array(u_history)
		self.time_history = np.array(time_history)

		if self.initial:
			np.save('results/'+self.case+'/PIDTracked_q_initial.npy',self.q_history)
			np.save('results/'+self.case+'/PIDTracked_q_dot_initial.npy',self.q_dot_history)
			np.save('results/'+self.case+'/PIDTracked_u_initial.npy',self.u_history)
			np.save('results/'+self.case+'/PIDTracked_time_intial.npy',self.time_history)

		else:
			iteration = 70
			np.save('results/'+self.case+'/PIDTracked_q_'+str(iteration)+'.npy',self.q_history)
			np.save('results/'+self.case+'/PIDTracked_q_dot_'+str(iteration)+'.npy',self.q_dot_history)
			np.save('results/'+self.case+'/PIDTracked_u_'+str(iteration)+'.npy',self.u_history)
			np.save('results/'+self.case+'/PIDTracked_time_'+str(iteration)+'.npy',self.time_history)




	def _targetQ(self,time):
		return self.ref_trajectory.eval(time,True)[3:15]
	def _targetQFull(self,time):
		return self.ref_trajectory.eval(time,True)

if __name__=="__main__":

	##### code to evaluate the intial guess
	# traj_guess = np.hstack((np.load('results/PID_trajectory/2/q_init_guess.npy'),np.load('results/PID_trajectory/2/q_dot_init_guess.npy')))
	# u_guess = np.load('results/PID_trajectory/2/u_init_guess.npy')
	# #This is the dt = 0.005 PID trajectory
	# traj_guess = np.hstack((np.load('results/PID_trajectory/2/q_history.npy'),np.load('results/PID_trajectory/2/q_dot_history.npy')))
	# traj_guess = np.load('results/PID_trajectory/3/x_init_guess.npy')
	# u_guess = np.load('results/PID_trajectory/3/u_init_guess.npy')
	# analyzer = analyzer('',dt = 0.05,method = "Euler",x_data = traj_guess, u_data = u_guess)
	# analyzer.calculation()
	# analyzer.animate()
	# analyzer.dynConstrViolation()
	# print('objective is',analyzer.objective(traj_guess,u_guess))
	# print(traj_guess[0,0])
	# print(traj_guess[-1,0])

	##### code to evaluate an optimized trajectory
	iteration = 10
	traj = np.load('results/16/run4/solution_x'+str(iteration) +'.npy')
	u = np.load('results/16/run4/solution_u'+str(iteration)+'.npy')

	# traj = np.load('results/17/solution_x'+str(iteration) +'.npy')
	# u = np.load('results/17/solution_u'+str(iteration)+'.npy')

	analyzer = analyzer('16',dt = 0.05,method = "BackEuler",x_data = traj, u_data = u)
	analyzer.calculation()
	analyzer.animate() #animate the trajectory
	# print('objective is',analyzer.objective(traj,u_))
	# print('initial torso x:',traj[0,0])
	# print('final torso x:',traj[-1,0])
	#### plot the progress of optimization
	####need to do special handling inside the class 
	#analyzer.perIterationObj()
	#analyzer.perIterGeneralConstrVio('enough_translation')
	#analyzer.perIterGeneralConstrVio('ankle_pose')
	#analyzer.perIterDynConstrVio()



	######## Track a trajectory with PID controller

	##------ initial trajectory -------##

	##### code to generate the PID tracked trajectory of initial guess(with large dt)
	# traj_guess = np.hstack((np.load('results/PID_trajectory/2/q_init_guess.npy'),np.load('results/PID_trajectory/2/q_dot_init_guess.npy')))
	# u_guess = np.load('results/PID_trajectory/2/u_init_guess.npy')
	# tracker = PIDTracker('14',True,traj_dt=0.05,x_data = traj_guess,u_data = u_guess, initial = True)
	# tracker.run()

	##### code to check the intial guess tracked by PID controller
	# traj_guess = np.hstack((np.load('results/14/PIDTracked_q_initial.npy'),np.load('results/14/PIDTracker_q_dot_initial.npy')))
	# u_guess = np.load('results/14/PIDTracker_u_initial.npy')
	# analyzer = analyzer('',dt = 0.005,method = "Euler",x_data = traj_guess, u_data = u_guess)
	# # analyzer.calculation()
	# # analyzer.animate()
	# print('objective is',analyzer.objective(traj_guess,u_guess))
	# print(traj_guess[0,0])
	# print(traj_guess[-1,0])


	## ----- an optimized trajectory ----- ##

	##### code to generate PID tracked trajectory of the optimized result(with large dt)
	# iteration = 70
	# traj_guess = np.hstack((np.load('results/15/solution_x'+str(iteration) +'.npy'),np.load('results/15/solution_x'+str(iteration) +'.npy')))
	# u_guess = np.load('results/15/solution_u'+str(iteration)+'.npy')
	# tracker = PIDTracker('15',True,traj_dt=0.05,x_data = traj_guess,u_data = u_guess, initial =False)
	# tracker.run()



	#print('objective is',analyzer.objective(traj_guess,u_guess))

	##### code to check optimized trajectory tracked by PID controller
	# iteration = 70
	# traj_guess = np.hstack((np.load('results/15/PIDTracked_q_'+str(iteration) +'.npy'),np.load('results/15/PIDTracked_q_dot_'+str(iteration) +'.npy')))
	# u_guess = np.load('results/15/PIDTracked_u_'+str(iteration)+'.npy')
	# analyzer = analyzer('',dt = 0.005,method = "BackEuler",x_data = traj_guess, u_data = u_guess)
	# print('objective is',analyzer.objective(traj_guess,u_guess))
	# print(traj_guess[0,0])
	# print(traj_guess[-1,0])

