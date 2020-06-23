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
class analyzer2:
	def __init__(self,x_data,u_data,case = '17',run = None,dt = 0.005,method = "Euler"):
		self.dt = dt
		q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
		q_dot0 = np.zeros((15,1))
		self.robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = dt, solver = 'cvxpy', augmented = True, extrapolation= True)
		self.case = case
		if run:
			self.run = run
			self.run_flag = True
		else:
			self.run_flag = False

		self.x = x_data
		self.u = u_data
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

	def perIterationObj(self,total_iters,prefix):
		objective_values = []
		iterations = []

		for i in range(total_iters):
			# if i == 0:
			# 	#add the initial guess's values
			# 	objective_values.append(self.objective(initial_x,initial_u))
			# 	iterations.append(0)
			# else:
			objective_values.append(np.load(prefix+str(i)+'.npy'))
			iterations.append(i)

		##plotting
		plt.plot(iterations,objective_values)
		plt.title('Objective Values over Iterations')
		plt.grid()
		plt.ylabel('Objective')
		plt.xlabel('Iterations')
		plt.show()
		return


	def perIterationConstrVio(self,total_iters, prefix):
		##dynamics constraint
		iterations = []
		dyn_constr_violations = []
		dyn_constr_violations_l1 = []
		for i in range(total_iters):
			iterations.append(i)
			if i == 0:
				constr_values = np.load(prefix+str(i)+'.npy')
			else:
				constr_values = np.load(prefix+str(i)+'.npy')
				#remove the first and last
				constr_values = constr_values[1:6879]
			print(np.shape(constr_values))
			#parse the constraint values
			violation = []
			N_of_dyn = 5400
			for i in range(N_of_dyn):
				violation.append(math.fabs(constr_values[i]))

			for i in range(181*8):
				if i%2 == 0:
					depth = constr_values[i+N_of_dyn]
					if depth > 1.0:
						violation.append(depth-1.0)
					if depth < -0.2:
						violation.append(-0.2-depth)
				else:
					angle = constr_values[i+N_of_dyn]
					if angle > 1.0:
						violation.append(angle-1.0)
					if angle < -1.0:
						violation.append(-1.0-angle)

			# if constr_values[N_of_dyn+181*8] > 0.4:
			# 	violation.append(0.0)
			# else:
			# 	violation.append(0.4 - constr_values[N_of_dyn+181*8])


			print(constr_values[6848])
			dyn_constr_violations.append(np.linalg.norm(violation))
			dyn_constr_violations_l1.append(np.max(np.absolute(violation)))

			print(dyn_constr_violations,dyn_constr_violations_l1)
		##plotting
		plt.plot(iterations,dyn_constr_violations,iterations,dyn_constr_violations_l1)
		plt.legend(['2-norm','Max Violation'])
		plt.title('Constraint Violations over Iterations')
		plt.grid()
		plt.ylabel('Constraint Violation')
		plt.xlabel('Iterations')
		plt.show()

		return

		
	def objective(self,x,u):
		"""
		This is problem specific
		"""
		scale = 1000.0
		small_C = 0.01
		#below is the transportation cost
		effort_sum = 0.0
		#print(self.N)
		for i in range(self.N):
			for j in range(12):
				#q_dot * u
				effort_sum = effort_sum + (x[i,j+18]*u[i,j])**2
		obj = effort_sum/(x[self.N -1 ,0] - x[0,0] + small_C)/scale

		return obj				

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
	x = np.load('results/17/run4/solution_x1.npy')
	u = np.load('results/17/run4/solution_u1.npy')
	analyzer = analyzer2(x,u,case = '17',run = '3',dt = 0.05,method = "BackEuler")
	analyzer.perIterationConstrVio(1,'temp_files/knitro_con')
	#analyzer.perIterationObj(11,'results/16/run4/knitro_obj')



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
	# iteration = 20
	# traj = np.load('results/17/run2/solution_x'+str(iteration) +'.npy')
	# u = np.load('results/17/run2/solution_u'+str(iteration)+'.npy')

	# # traj = np.load('results/17/solution_x'+str(iteration) +'.npy')
	# # u = np.load('results/17/solution_u'+str(iteration)+'.npy')

	# analyzer = analyzer('17',dt = 0.05,method = "BackEuler",x_data = traj, u_data = u)
	# analyzer.calculation()
	# analyzer.animate() #animate the trajectory
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

