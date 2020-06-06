"""
This file includes the class of methods for visualizing and analyzing the results from 
a trajectory optimizer
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
class analyzer:
	def __init__(self,case = '11-2',dt = 0.005,method = "Euler"):
		self.dt = dt
		q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
		q_dot0 = np.zeros((15,1))
		self.robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = dt, solver = 'cvxpy', augmented = True)
		self.case = case
		self.u = np.load('results/'+case+'/solution_u.npy')
		self.x = np.load('results/'+case+'/solution_x.npy')
		(self.N,_) = np.shape(self.x)
		self.method = method
		self.iterations = np.arange(self.N)
		self.times = self.iterations*self.dt
		#slow down 20x

		#settings for animating the trajectory
		self.vis_dt = 0.005*40
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
			if i < self.N - 1:
				if self.method == 'Euler':
					#print(np.shape(a),np.shape(self.x))
					p_error = self.x[i+1,0:15]-self.x[i,0:15]-self.dt*self.x[i,15:30]
					v_error = self.x[i+1,15:30]-self.x[i,15:30]-self.dt*a
					#print(p_error,v_error)
					print('accel',a)
					print('force',force)
					print(np.linalg.norm(np.concatenate((p_error,v_error))),np.linalg.norm(self.x[i,:]),np.linalg.norm(self.x[i+1,:]-self.x[i,:]) ,np.linalg.norm(p_error))
					self.violations.append(np.linalg.norm(np.concatenate((p_error,v_error))))
					# #print(np.linalg.norm(self.x[i+1,:]-self.x[i,:]-self.dt*a),np.linalg.norm(self.x[i,:]))
					# print(a)
					# print(self.x[i,:])
					# print(a-self.x[i,:])
		return 

	def animate(self):
		current_time = 0.0
		counter = 0
		vis.addText('time','time: '+str(0.0))
		vis.show()
		time.sleep(1)
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
		plt.plot(self.times[0:self.N-1],self.violations)
		plt.title('Dynamics Constraint Violation')
		plt.grid()
		plt.ylabel('Unscaled Constraint Violation 2-norm')
		plt.xlabel('Time (s)')
		plt.show()
		return

	def plotObjective(self):
		"""
		This is problem specific
		"""
		

		return								

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

class PIDTracker:
	def __init__(self,case,visualization = True, dt = 0.005):
		self.visualization = visualization
		self.dt = dt
		q0 = np.array(configs.q_staggered_augmented)[np.newaxis].T
		q_dot0 = np.zeros((15,1))
		self.robot = robosimianSimulator(q = q0, q_dot = q_dot0, dt = dt, solver = 'cvxpy', augmented = True)
		self.case = case
		self.u = np.load('results/'+case+'/solution_u.npy')
		self.x = np.load('results/'+case+'/solution_x.npy')

		self.kp = np.array([1000.0,1000.0,1000.0]*4)
		self.ki = np.array([2.0]*12)
		self.kd = np.array([10.0,8.0,10.0]*4)

		(self.N,_) = np.shape(self.x)
		self.iterations = np.arange(self.N)
		self.times = self.iterations*self.dt
		#slow down 20x
		self.ref_trajectory = trajectory.Trajectory(times = self.times ,milestones = self.x.tolist())
		self.ref_torque = trajectory.Trajectory(times = self.times, milestones = self.u.tolist())
		#settings for animating the trajectory
		self.vis_dt = 0.005*40
		self.force_scale = 0.001 #200N would be 0.2m

		self.world = self.robot.getWorld()
		vis.add("world",self.world)

	def run(self):
		self.ankle_position_list = []
		self.force_list = []

		if self.visualization:
			vis.show()
			vis.addText('time','time: '+str(0.0))

		#reset the robot
		self.robot.reset(self.x[0,0:15],self.x[0,15:30])
		#start simulation
		error = np.array([0.0]*12)
		last_error = np.array([0.0]*12)
		accumulated_error = np.array([0.0]*12)
		time.sleep(2.0)
		simulation_time = 0.0
		#start_time = time.time()

		#record these
		q_history = []
		q_dot_history = []
		u_history = []
		time_history = []


		while vis.shown() and (simulation_time < self.times[-1]+0.0001):
			#loop_start_time = time.time()
			vis.lock()
			#simulation_time = time.time() - start_time
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
			tau = tau + np.array(self.ref_torque.eval(simulation_time,True))
			tau = np.clip(tau,-300.0,300.0) 

			#record 
			q_history.append(self.robot.getConfig().tolist())
			q_dot_history.append(self.robot.getVel().tolist())
			u_history.append(tau.tolist())
			time_history.append(simulation_time)

			simulation_time += self.dt
			#vis.clearText()
			vis.addText('time','time: '+str(simulation_time))
			# print('time',simulation_time)
			# print('desired q',desired_q)
			# print('current q',current_q)
			# print('error',error)
			# print('u:',u)

			force,a = self.robot.simulateOnce(tau,continuous_simulation = True)
			ankle_positions = self.robot.robot.get_ankle_positions(full = True)
			self.force_list.append(force)
			self.ankle_position_list.append(ankle_positions)
			vis.unlock()
			time.sleep(0.001)

		while vis.shown():
			time.sleep(0.1)
		vis.kill()

		#save stuff
		self.q_history = np.array(q_history)
		self.q_dot_history = np.array(q_dot_history)
		self.u_history = np.array(u_history)
		self.time_history = np.array(time_history)
		np.save('results/'+self.case+'/PIDTracker_q.npy',self.q_history)
		np.save('results/'+self.case+'/PIDTracker_q_dot.npy',self.q_dot_history)
		np.save('results/'+self.case+'/PIDTracker_u.npy',self.u_history)
		np.save('results/'+self.case+'/PIDTracker_time.npy',self.time_history)


	def _targetQ(self,time):
		return self.ref_trajectory.eval(time,True)[3:15]

if __name__=="__main__":
	#analyzer = analyzer('8',dt = 0.005,"Euler")
	#start_time = time.time()
	#analyzer.calculation()
	#print('calculation done, took',time.time() - start_time)
	#analyzer.animate()
	#analyzer.dynConstrViolation()

	#########
	#tracker = PIDTracker('8',True,dt=0.005)
	#tracker.run()