import numpy as np
import random
import tensorflow as tf
from keras import layers
from collections import deque

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        out = layers.Dense(400, activation="relu")(inputs)
        out = layers.Dense(300, activation="relu")(out)
        outputs = layers.Dense(self.action_size, activation="tanh")(out)
        outputs = outputs * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001))
        return model

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        state_input = layers.Input(shape=(self.state_size,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        action_input = layers.Input(shape=(self.action_size,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(400, activation="relu")(concat)
        out = layers.Dense(300, activation="relu")(out)
        outputs = layers.Dense(1, activation="linear")(out)

        model = tf.keras.Model([state_input, action_input], outputs)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.002))

        return model

class DDPG:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.actor = Actor(state_size, action_size, action_low, action_high)
        self.critic = Critic(state_size, action_size)
        self.buffer = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.001

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        minibatch = random.sample(self.buffer, batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Critic update
        target_actions = self.actor.model.predict_on_batch(next_states)
        future_rewards = self.critic.model.predict_on_batch([next_states, target_actions])
        future_rewards = future_rewards.squeeze()
        targets = rewards + self.gamma * (1 - dones) * future_rewards
        with tf.GradientTape() as tape:
            current_Q_values = self.critic.model([states, actions])
            critic_loss = tf.keras.losses.MSE(current_Q_values, targets)
        critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.model.optimizer.apply_gradients(zip(critic_grad, self.critic.model.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            actions_pred = self.actor.model(states)
            critic_value_pred = self.critic.model([states, actions_pred])
            actor_loss = -tf.math.reduce_mean(critic_value_pred)
        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.model.optimizer.apply_gradients(zip(actor_grad, self.actor.model.trainable_variables))

        self.soft_update(self.actor.model, self.critic.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def act(self, state, noise_scale=0.1):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor.model.predict(state)[0]
        noise = noise_scale * np.random.normal(size=self.action_size)
        action = np.clip(action + noise, self.action_low, self.action_high)
        return action

    def save_weights(self, actor_path, critic_path):
        self.actor.model.save_weights(actor_path)
        self.critic.model.save_weights(critic_path)

    def load_weights(self, actor_path, critic_path):
        self.actor.model.load_weights(actor_path)
        self.critic.model.load_weights(critic_path)
