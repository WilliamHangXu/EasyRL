import gym
from reinforce_xuh9 import RFAgent


def test_agent():
    episode_rewards = []

    # Initialize the environment and the agent
    env = gym.make('CartPole-v0', render_mode="human")
    agent = RFAgent(n_actions=env.action_space.n, n_states=env.observation_space.shape[0])

    # Load the saved weights
    agent.load_model('rf_weights.h5')

    # Test the model
    for episode in range(10):
        observation = env.reset()[0]
        episode_reward = 0
        num_steps = 1
        while True:
            print("Step {}".format(num_steps))
            # Visualize the environment
            env.render()
            # Choose an action based on observation
            action = agent.choose_action(observation)
            # Apply the action and get new measurements
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            num_steps += 1
            if terminated or truncated:
                num_steps = 1
                print('Test Episode: {}  Reward: {}'.format(episode + 1, episode_reward))
                break
        episode_rewards.append(episode_reward)
    print('Episode rewards:')
    print(episode_rewards)
    env.close()


if __name__ == "__main__":
    test_agent()