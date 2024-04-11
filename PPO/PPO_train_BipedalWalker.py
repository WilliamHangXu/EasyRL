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
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

#################### Hyperparameters ##############################
max_ep = 200000
lr = 0.0003
epochs = 10
batch = 512
gamma = 0.99
clip = 0.2
lam = 0.90
log_std = -0.5 * np.ones(action_dim, dtype=np.float32)

#################### Training Setup ##############################
# Create the agent
ppo = PPO(state_dim, action_dim, lr, epochs, gamma, clip, lam, log_std)
# Reset the environment
state = env.reset()[0]
state = np.reshape(state, [1, state_dim])
# Episode reward
ep_reward = 0
# The average of the previous 100 episode rewards
avg_reward = 0
# Number of episodes trained
num_ep = 0
# Stores episode rewards.
ep_rewards = []
# Stores average episode rewards.
avg_rewards = []
# Stores actor loss and critic loss
actor_losses = []
critic_losses = []

#################### Training ##############################
# Records starting time
start = timer()
# Break flag. If a certain amount of training is done, the program needs to break two loops.
flag = False

'''
I am using "while True" instead of "for i in range(max_ep)" because one iteration does not necessarily correspond to one 
episode. The inner for loop collects a number of steps in a trajectory without considering episode termination. For
example, a batch might contain several episodes, or one huge incomplete episode. This was something that I failed to
consider before.
'''
while True:
    # Clear buffers
    ppo.reset_buffer()
    for t in range(batch):
        # The actor selects an action and calculates the probability density of taking that action.
        action, logpdf = ppo.act(state)
        # Collects step information
        next_state, reward, terminated, truncated, info = env.step(action)
        # Both terminated and truncated means that an episode is over. So we can sum them up.
        done = terminated or truncated
        # Stores step information.
        ppo.store(state, action, np.reshape(next_state, [1, state_dim]), reward, done, logpdf[0])
        # Set current state to next state
        state = np.reshape(next_state, [1, state_dim])
        # Increment episode reward by the reward of this step
        ep_reward += reward

        # If an episode is done...
        if done:
            # Increment the episodes counter
            num_ep += 1
            # Stores rewards
            ep_rewards.append(ep_reward)
            avg_reward = np.mean(ep_rewards[-100:])
            avg_rewards.append(avg_reward)
            # Print info
            print("episode: {}/{}, reward: {}, avg: {:.2f}".format(num_ep, max_ep, ep_reward, avg_reward))
            # If it reaches the cap, stop training, save model weights, and print failure info
            if num_ep >= max_ep:
                ppo.save()
                print('Max episodes reached. Did not achieve your goal.')
                flag = True
                break
            # Reset episode reward
            ep_reward = 0
            # Reset environment
            state = env.reset()[0]
            state = np.reshape(state, [1, state_dim])
    if flag:
        break
    # Train the model and record actor loss and critic loss.
    a_loss, c_loss = ppo.train()
    actor_losses.append(np.mean(a_loss))
    critic_losses.append(np.mean(c_loss))

    # If the expected performance is acheived, stop training, save model weights, and print info
    # My agent has high variance, so reaching reward 300 does not mean that training is finished.
    # I am using my own terminating criteria instead.
    if avg_reward >= 300:
        ppo.save()
        print('Solved! Weights Saved.')
        break
# Record end time
end = timer()
# Close the environment.
env.close()

#################### Graph and Data ##############################
# Plot the rewards
print("Training time: {}s".format(end-start))
plt.figure()
plt.plot(ep_rewards, label='episode rewards')
plt.plot(avg_rewards, label='avg return of previous 100 episodes')
plt.xlabel('Episodes')
plt.legend()
plt.savefig('rewards.png', bbox_inches='tight', dpi=300)

# Plot the losses
plt.figure()
plt.plot(actor_losses, label='actor_loss')
plt.plot(critic_losses, label='critic_loss')
plt.xlabel('Iteration')
plt.legend()
plt.savefig('loss.png', bbox_inches='tight', dpi=300)

# Record episode rewards, actor loss and critic loss
with open('data.txt', 'w') as file:
    file.write('========== Training Data ==========\n')
    file.write('\n')
    file.write('Time to complete training: {}s\n'.format(end - start))
    file.write('Number of episodes to complete training: {}\n'.format(num_ep))
    file.write('\n')
    file.write('#episode    reward                    actor_loss                    critic_loss\n')
    for i in range(len(ep_rewards)):
        file.write('{:<12}'.format(i + 1))
        file.write('{:<26}'.format(ep_rewards[i]))
        file.write('{:<30}'.format(actor_losses[i]))
        file.write('{}'.format(critic_losses[i]))
        file.write('\n')
