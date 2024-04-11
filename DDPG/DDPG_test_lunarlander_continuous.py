import gym
from ddpg_xuh9 import DDPG

# Environment setup
env = gym.make('LunarLanderContinuous-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_low = env.action_space.low
action_high = env.action_space.high

# Initialize agent
agent = DDPG(state_size, action_size, action_low, action_high)
actor_weights_path = 'weights/final_actor.h5' # Change to your model path if different
critic_weights_path = 'weights/final_critic.h5' # Change to your model path if different

# Load the agent weights
agent.load_weights(actor_weights_path, critic_weights_path)

# Test the agent
num_episodes = 100

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state, noise_scale=0.0) # Set noise to 0 for testing
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
        total_reward += reward
    print("Episode: {}, total reward: {:.2f}".format(ep, total_reward))

env.close()
