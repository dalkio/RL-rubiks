import random
import numpy as np
from inspect import isclass
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, environment_type, action_type,
                 lr=1e-3, hidden_size=24, gamma=0.95, memory_size=1e6, batch_size=20, 
                 exploration_max=1.0, exploration_min=1e-2, exploration_decay=0.995,
                 verbose=1):
        # Exploration values
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = self.exploration_max
        # Action type for the environment
        assert isclass(action_type)
        self.action_type = action_type
        # Environment properties
        self.environment_type = environment_type
        self.environment = None
        self.state_space = None
        self.action_space = None
        # Q stack memory
        assert memory_size >= batch_size
        self.memory = deque(maxlen=int(memory_size))
        # Discount factor
        self.gamma = gamma
        # Deep model parameters
        self.lr = lr
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.model = None
        # Verbose
        self.verbose = verbose
        # Initialization
        self._initialize()

    def _create_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_shape=(self.state_space,), activation="relu"))
        model.add(Dense(self.hidden_size, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model
    
    def _initialize(self):
        self.environment = self.environment_type(verbose=self.verbose>1)
        self.state_space = len(self.environment.state.flatten())
        self.action_space = len(self.environment.actions)
        self.model = self._create_model()
    
    def _exploration_update(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(np.ravel(q_values))

    def experience_replay(self):
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.gamma * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self._exploration_update()
        
    def train(self, episodes=100, max_step=1000, display_frequence=100):
        for episode in range(episodes):
            self.environment.reset()
            state_flat = np.reshape(self.environment.state, [1, self.state_space])
            for step in range(max_step):
                action_index = self.predict(state_flat)
                action = self.action_type(self.environment.actions[action_index])
                reward, terminal = self.environment.step(action)
                state_next_flat = np.reshape(self.environment.state, [1, self.state_space])
                self.remember(state_flat, action_index, reward, state_next_flat, terminal)
                state_flat = state_next_flat
                if self.verbose:
                    if terminal:
                        print("Run: {0}, exploration: {1}, score: {2}".format(
                                str(episode), str(self.exploration_rate), str(step)
                            )
                        )
                        break
                    if step%display_frequence == 0:
                        print("Step: {0}, reward: {1}".format(
                                step, reward
                            )
                        )
                self.experience_replay()