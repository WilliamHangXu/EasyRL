import gym
import numpy as np
from reinforce_xuh9 import RFAgent
import matplotlib.pyplot as plt
from timeit import default_timer as timer


# The array that stores loss for each episode.
loss_array = []
# The array that stores rewards for each episode.
episode_rewards = []

def train_agent():
    # Create the gym environment
    env = gym.make('CartPole-v0')
    # Initialize the agent
    agent = RFAgent(n_actions=env.action_space.n, n_states=env.observation_space.shape[0])

    # Specify the number of episodes the model will be trained.
    num_episodes = 10000

    # For each training episode...
    for episode in range(num_episodes):

        # Reset the environment and get the initial observation
        observation = env.reset()[0]
        episode_reward = 0
        num_steps = 1

        # While the episode is still running, for each time step...
        while True:
            print('Step {}'.format(num_steps))
            # Choose an action based on the observation of current state
            action = agent.choose_action(observation)
            # Apply the action, then the environment transits to the next state. From the transition, we get the new
            # state and reward.
            new_observation, reward, terminated, truncated, info = env.step(action)
            # Store the original state observation, the action performed in this state, and the reward we got from it.
            agent.store_trajectory(observation, action, reward)
            # Update the observation.
            observation = new_observation
            # Increment the total reward by the reward we got from this action.
            episode_reward += reward
            num_steps += 1

            # If the episode ends...
            if terminated or truncated:
                num_steps = 1
                # Add the reward of this episode to the array
                episode_rewards.append(episode_reward)
                # Train the agent. Get the loss and store it.
                loss = agent.learn()
                loss_array.append(loss)
                print('Episode: {}  Reward: {}'.format(episode, episode_reward))
                # Start a new episode
                break

        # Check if the condition of averaging 195 reward over 100 consecutive episodes is met
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            print("Average return of last 100 episodes: ", avg_100)
            if avg_100 >= 195:
                print("Solved after {} episodes!".format(episode))
                # If the condition is met, stop training.
                break

    # Save model weights after training is complete
    agent.save_model('rf_weights.h5')


if __name__ == "__main__":

    # Train the agent and record the starting and ending time.
    start = timer()
    train_agent()
    end = timer()
    # Create an array to store the average reward of the previous 100 episodes. The first 100 elements are NaN.
    avg_100_array = np.empty_like(episode_rewards)
    avg_100_array[:] = np.nan
    for i in range(100, len(episode_rewards)):
        avg_100_array[i] = np.mean(episode_rewards[i - 100:i])

    # Store the episode rewards and losses into a .txt file.
    with open('data.txt', 'w') as file:
        file.write('TRAINING DATA\n')
        file.write('\n')
        file.write('Time to complete training: {}s\n'.format(end-start))
        file.write('\n')
        file.write('#episode    reward        loss\n')
        for i in range(len(episode_rewards)):
            file.write('{:<12}'.format(i+1))
            file.write('{:<14}'.format(episode_rewards[i]))
            file.write('{}'.format(float(loss_array[i].numpy())))
            file.write('\n')

    # Plot the graphs and save them to the directory.
    plt.figure()
    plt.plot(avg_100_array, label='avg return of previous 100 episodes')
    plt.plot(episode_rewards, label='episode rewards')
    plt.xlabel('Episodes')
    plt.legend()
    plt.savefig('rewards.png', bbox_inches='tight', dpi=300)

    plt.figure()
    plt.plot(loss_array, label='loss')
    plt.xlabel('Episodes')
    plt.legend()
    plt.savefig('loss.png', bbox_inches='tight', dpi=300)