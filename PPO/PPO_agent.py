import numpy as np
import tensorflow as tf
import copy

'''
Reference:
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
https://pylessons.com/BipedalWalker-v3-PPO
'''

class PPO:
    # Constructor
    def __init__(self, state_dim, action_dim, lr, epochs, gamma, clip, lam, log_std):
        # Action dimension
        self.action_dim = action_dim
        # State dimension
        self.state_dim = state_dim
        # Learning rate for both actor and critic
        self.lr = lr
        # Number of epochs of training for each batch
        self.epochs = epochs
        # Discount factor
        self.gamma = gamma
        # Probability ratio clipping parameter
        self.clip = clip
        # GAE parameter lambda
        self.lam = lam
        # log standard deviation and standard deviation for sampling
        self.log_std = log_std
        self.std = np.exp(self.log_std)
        # Actor and critic
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        # Set up buffer
        self.reset_buffer()

    # Set up / clear buffer
    def reset_buffer(self):
        self.states_buffer = []
        self.actions_buffer = []
        self.next_states_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.logpdfs_buffer = []

    # Store state transition information to buffers
    def store(self, state, action, next_state, reward, done, logpdfs):
        self.states_buffer.append(state)
        self.actions_buffer.append(action)
        self.next_states_buffer.append(next_state)
        self.rewards_buffer.append(reward)
        self.dones_buffer.append(done)
        self.logpdfs_buffer.append(logpdfs)

    # Construct the actor
    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_dim=self.state_dim,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            tf.keras.layers.Dense(self.action_dim, activation='tanh')
        ])
        model.compile(loss=self.actor_loss, optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        return model

    # Construct the critic
    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_dim=self.state_dim,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss=self.critic_loss, optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        return model

    # Loss function for actor
    # y_true: packed information, including gae result, actions, and the probability densities of those actions
    # y_pred: output of actor given states
    '''
    The paper says that update is performed for several epochs for one batch. I thought that we should update pi_old 
    after every epoch, but it turns out that pi_old should remain unchanged. In each epoch, the new actor is compared
    against the old actor that's before the first epoch/
    '''
    def actor_loss(self, y_true, y_pred):
        # Unpack y_true. To use this loss function in the fit function of tensorflow, this function can only have two
        # parameters, y_true and y_pred. So we need to put all things in y_true.
        advantages, actions, logpdfs_old, = y_true[:, :1], y_true[:, 1:1 + self.action_dim], y_true[:, 1 + self.action_dim]
        # Calculate the probability density of taking those actions with the new actor
        logp = self.gaussian_likelihood_tf(actions, y_pred)
        # Calculate probability ratio
        ratio = tf.exp(logp - logpdfs_old)
        # Calculate the first loss term.
        surr1 = ratio * advantages
        # Calculate the second loss term
        surr2 = tf.where(advantages > 0, (1.0 + self.clip) * advantages, (1.0 - self.clip) * advantages)
        surr2 = tf.cast(surr2, dtype=tf.float32)
        # The objective is the minimum value of surr1 and surr2. We want to maximize this value, so we minimize its
        # negative value, which is the actor_loss that this function returns
        actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        return actor_loss

    # Loss function for critic
    # y_true: the discounted returns for each state
    # y_pred output of critic given states
    def critic_loss(self, y_true, y_pred):
        # The loss is simply the squared error
        critic_loss = tf.reduce_mean((y_true - y_pred) ** 2)
        return critic_loss

    # Calculates the probability density of actions given a gaussian distribution with pred as mean and self.std as
    # standard deviation
    # This function returns a tensor, which is useful in the loss function. Numpy arrays doesn't work in that function.
    def gaussian_likelihood_tf(self, actions, pred):
        # This is the logged version of gaussian distribution's pdf
        logpdf = - (actions - pred) ** 2 / (2 * self.std ** 2) - self.log_std - 0.5 * tf.math.log(2 * np.pi)
        result = tf.reduce_sum(logpdf, axis=1)
        return result

    # Calculates the probability density of actions given a gaussian distribution with pred as mean and self.std as
    # standard deviation
    # This function returns a Numpy array.
    def gaussian_likelihood_np(self, actions, pred):
        # This is the logged version of gaussian distribution's pdf
        logpdf = - (actions - pred) ** 2 / (2 * self.std ** 2) - self.log_std - 0.5 * np.log(2 * np.pi)
        result = np.sum(logpdf, axis=1)
        return result

    # Given a state, selects an action and calculates its probability density
    def act(self, state):
        # Let actor predict an action
        action_mean = self.actor.predict(state)
        # Add noise to the action
        # I do not understand why this works. The paper says we need to sample action from a gaussian distribution,
        # but here I use uniform noise.
        action = action_mean + np.random.uniform(-1.0, 1.0, size=self.action_dim) * self.std
        # I tried these two sampling methods and none of them worked:
        # action = pred + np.random.normal(self.action_dim) * self.std
        # action = np.random.normal(action_mean, self.std, size=4)
        # Clip the action so that it is within range
        action = np.clip(action, -1.0, 1.0)
        # Calculates the log probability density of taking the action
        logpdf = self.gaussian_likelihood_np(action, action_mean)
        return action[0], logpdf

    # Calculates advantages and targets for critic.
    def gae(self, values, next_values):
        # Calculate deltas for each state. We do not take the terminal state into account
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(self.rewards_buffer, self.dones_buffer, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        # Calculate advantages for each state.
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - self.dones_buffer[t]) * self.gamma * self.lam * gaes[t + 1]
        # Calculates the targets for critic. Advantage is basically discounted return minus critic output. We obtain
        # discounted return by adding critic output to advantages.
        # I also tried directly calculating discounted returns but it did not work.
        value_target = gaes + values
        # Normalize advantages
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(value_target)

    # Main training function
    def train(self):
        # Reshape buffer information for training
        states = np.vstack(self.states_buffer)
        next_states = np.vstack(self.next_states_buffer)
        actions = np.vstack(self.actions_buffer)
        logpdfs = np.vstack(self.logpdfs_buffer)
        # Get value predictions from critic
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)
        # Calculates advantages and critic targets
        advantages, value_target = self.gae(np.squeeze(values), np.squeeze(next_values))
        # Store all information into y_true
        y_true = np.hstack([advantages, actions, logpdfs])
        # Train actor and critic and obtain loss
        # I also tried training manually, i.e. using tf.GradientTape blocks. It did not work.
        a_loss = self.actor.fit(states, y_true, epochs=self.epochs, verbose=0).history['loss']
        c_loss = self.critic.fit(states, value_target, epochs=self.epochs, verbose=0).history['loss']
        return a_loss, c_loss

    # Load weights
    def load(self):
        self.actor.load_weights('actor_weights.h5')
        self.critic.load_weights('critic_weights.h5')

    # Save weights
    def save(self):
        self.actor.save_weights('actor_weights.h5')
        self.critic.save_weights('critic_weights.h5')