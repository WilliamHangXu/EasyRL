import gym
import numpy as np
import tensorflow as tf
from ppo_xuh9 import PPO
import matplotlib.pyplot as plt
from timeit import default_timer as timer
# Disable GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Disable eager execution to speed up training
tf.compat.v1.disable_eager_execution()

#################### Environment Setup ##############################
env_name = 'BipedalWalker-v3'
env = gym.make(env_name, render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

#################### Hyperparameters ##############################
max_ep = 20000
lr = 0.0003
epochs = 10
batch = 512
gamma = 0.99
clip = 0.2
lam = 0.90
log_std = -0.5 * np.ones(action_dim, dtype=np.float32)

#################### Testing Setup ##############################
ppo = PPO(state_dim, action_dim, lr, epochs, gamma, clip, lam, log_std)
ppo.load()
ep_reward = 0

#################### Testing ##############################
for i in range(10):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_dim])
    while True:
        # I'm not using random noise for testing purpose
        action = ppo.actor.predict(state)[0]
        state, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated
        state = np.reshape(state, [1, state_dim])
        if done:
            print("episode: {}/10, reward: {}".format(i+1, ep_reward))
            ep_reward = 0
            break

env.close()