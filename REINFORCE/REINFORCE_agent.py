import tensorflow as tf
import numpy as np
from keras.layers import Dense


class RFAgent:
    def __init__(self, n_actions, n_states, lr=0.01, gamma=0.95):
        # Dimension of action space
        self.n_actions = n_actions
        # Dimension of state space
        self.n_states = n_states
        # Learning rate, alpha
        self.lr = lr
        # Discount factor, gamma
        self.gamma = gamma
        # Three arrays that store a trajectory's observations, actions, and rewards.
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        # Building the neural network
        self.build_model()

    def build_model(self):
        # Defines the structure of the neural network.
        self.model = tf.keras.Sequential([
            Dense(9, input_shape=(self.n_states,), activation='relu'),
            Dense(9, activation='relu'),
            Dense(self.n_actions, activation='softmax')
        ])
        # Compiles the model with optimizer.
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

    def choose_action(self, observation):
        # Given an observation of a state, use the model to calculate the weights (how desirable they are) of the two
        # actions available.
        act_weights = self.model.predict(np.array(observation[np.newaxis, :]))
        # Choose an action according to the weights.
        action = np.random.choice(range(self.n_actions), p=act_weights.ravel())
        return action

    def store_trajectory(self, observation, action, reward):
        # This function stores observations actions, and rewards of a trajectory into respective containers.
        self.episode_observations.append(observation)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def learn(self):
        # Compute discounted and normalized rewards
        final_reward = self.get_final_reward()

        # Compute the discount factors that will be applied to the weights
        disc = np.zeros_like(final_reward)
        for i in range(len(disc)):
            disc[i] = self.gamma ** i

        with tf.GradientTape() as tape:
            # Compute action probabilities using the policy neural network
            action_probs = self.model(np.vstack(self.episode_observations), training=True)
            # Compute log probabilities
            log_probs = tf.math.log(action_probs)
            # Select the log probabilities of chosen actions
            picked_log_probs = tf.gather(log_probs, self.episode_actions, batch_dims=1, axis=1)
            # Compute loss
            loss = -tf.reduce_mean(final_reward * picked_log_probs * disc)

        # Compute gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply the gradient to the optimizer so that the weights are updated
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Reset episode data
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        return loss

    def get_final_reward(self):
        # Create an empty array the same size of episode_rewards
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)

        # Calculate the discounted rewards
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        # Normalize the rewards. Probably not necessary.
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def save_model(self, path):
        # Export model weights to a .h5 file
        self.model.save_weights(path)

    def load_model(self, path):
        # Import model weights from a .h5 file
        self.model.load_weights(path)
