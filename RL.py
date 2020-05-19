import gym
from gym import spaces
from stable_baselines import PPO2
from robosimian_GM_simulation import robosimianSimulator
from stable_baselines.common.env_checker import check_env
import numpy as np
import configs
import os
import time
import math
class CustomEnv(gym.Env):
	def __init__(self):
		super(CustomEnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.dt = 0.02
		q_2D = np.array([0,0.915,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6])[np.newaxis].T
		q_dot_2D = np.array([0.0]*15)[np.newaxis].T
		self.robot = robosimianSimulator(q_2D,q_dot_2D,dt =self.dt,solver = 'cvxpy', augmented = False, RL = True)
		self.action_scale = 100.0
		#scaled * action_scale = FT
		self.action_space = spaces.Box(low = np.array([-1.0]*12), high = np.array([1.0]*12),dtype='float')
		self.observation_space = spaces.Box(low=np.array([-1000,-1000]+[-1000]*13 + [-300]*23), \
			high=np.array([1000,1000]+[1000]*13+[300]*23),dtype='float')
		self.default_q = np.array([0,0.915,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6])
		self.default_qdot = np.array([0]*15)
		#self.viewer = self.robot.getWorld
		self.count = 0
	def step(self, action):
		print('-------')
		print('current_iterations:',self.count)
		print('-------')

		u = action*self.action_scale
		u_prime = []
		for ele in u:
			u_prime.append(ele.item())
		u = np.array(u_prime)
		observation = self.robot.simulateOnceRL(u)
		done = False
		addtion_r = 0.0
		if observation[1] <  0.1:
			done = True
		if observation[1] >  1.5:
			done = True
		if observation[2] <  -1.2:
			done = True
		if observation[2] >  1.2:
			done = True
		if math.fabs(observation[31]) >  1.2:
			done = True
		if math.fabs(observation[33]) >  1.2:
			done = True
		if math.fabs(observation[35]) >  1.2:
			done = True
		if math.fabs(observation[37]) >  1.2:
			done = True

		if done:
			addtion_r = -3.0
		info = {}
		#reward = self.rewardFunc(np.ravel(observation),u)
		reward = self.rewardStand(np.ravel(observation),u)
		self.count = self.count + 1
		return np.ravel(observation), reward+addtion_r, done, info

	def rewardFunc(self,obs,act):

		effort_C = 1e-3
		effort_reward = -effort_C*np.linalg.norm(act)**2

		torso_height_reward = -0.01*(obs[1]-0.95)**2
		torso_angle_reward = -0.5*(obs[2])**2
		if obs[15] >= 0:
			forward_progress_reward = 1.0*obs[15]**2
		else:
			forward_progress_reward = -1.0*obs[15]**2

		ankle_angle_reward = 0.0

		for i in range(4):
			ankle_angle_reward = ankle_angle_reward - 0.2*(obs[30+2*i+1])**2

		return forward_progress_reward+torso_angle_reward+torso_height_reward+effort_reward+ankle_angle_reward

	def rewardStand(self,obs,act):

		return -np.linalg.norm(obs[15:30])**2-(obs[1]-0.915)**2 - (obs[0]-0.0)**2 - (obs[2]-0.0)**2 

	def reset(self):

		obs = self.robot.reset(self.default_q,self.default_qdot)
		return obs  # reward, done, info can't be included
	def render(self):
		return

	def close (self):
		return

if __name__=="__main__":
	start_time = time.time()
	env = CustomEnv()
	#check_env(env)
	model = PPO2('MlpPolicy', env,tensorboard_log ="RL_results/5")
	#model = PPO2.load("RL_results/test4")
	model.learn(total_timesteps=150000)
	log_dir = "RL_results/"
	model.save("RL_results/test5")

	print('Total time:',time.time() - start_time)
	# stats_path = os.path.join(log_dir, "stats.pkl")
	# env.save(stats_path)