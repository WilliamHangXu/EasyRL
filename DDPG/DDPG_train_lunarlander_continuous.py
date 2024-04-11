import gym
from ddpg_xuh9 import DDPG
import h5py
import os

# Make sure the necessary directories exist
if not os.path.exists('weights'):
    os.makedirs('weights')

# Environment setup
env = gym.make('LunarLanderContinuous-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_low = env.action_space.low
action_high = env.action_space.high

# Training setup
agent = DDPG(state_size, action_size, action_low, action_high)
num_episodes = 1000  # A typical amount of episodes for training
batch_size = 64  # Common batch size for training

# Main loop
for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.add_to_replay_buffer(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train
        agent.train(batch_size)

    print("Episode: {}, total reward: {:.2f}".format(ep, total_reward))

    # Save weights every 50 episodes
    if ep % 50 == 0:
        agent.save_weights('weights/actor.h5', 'weights/critic.h5')
        print("Weights saved at episode {}".format(ep))

# Final save
agent.save_weights('weights/final_actor.h5', 'weights/final_critic.h5')
print("Final weights saved")

# Close the environment
env.close()
