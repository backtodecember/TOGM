import gym
from gym import spaces
from stable_baselines import PPO2
from robosimian_GM_simulation import robosimianSimulator
from stable_baselines.common.env_checker import check_env
import numpy as np
import configs
import os
import time
import RL
from klampt import vis
q_2D = np.array([0,0.915,0,-0.9708,0,-0.6,2.1708,0,-0.6,-0.9708,0,-0.6,2.1708,0,-0.6])[np.newaxis].T
q_dot_2D = np.array([0.0]*15)[np.newaxis].T
robot = robosimianSimulator(q_2D,q_dot_2D,dt = 0.01,solver = 'cvxpy', augmented = False, RL = True)
world = robot.getWorld()
#load the model
log_dir = "RL_results/"
model = PPO2.load("RL_results/test6")

env = RL.CustomEnv()
obs = env.reset()
vis.add("world",world)
vis.show()
time.sleep(5)

while vis.shown():
	for i in range(1000):
		vis.lock()
		action, _states = model.predict(obs)
		action = action*100.0
		robot.simulateOnceRL(action)
		vis.unlock()
		time.sleep(0.01)

time.sleep(0.5)
vis.kill()