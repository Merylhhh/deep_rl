import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ─────────────────────────────────────────
#  Réseau de neurones Q
# ─────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
#  Replay Buffer (mémoire d'expériences)
# ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────
#  Agent DQN
# ─────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_size, action_size,
                 lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update=10):

        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.step_count    = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network      = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory    = ReplayBuffer()
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        """Politique ε-greedy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Q(s,a) courant
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q cible (réseau target figé)
        with torch.no_grad():
            next_q     = self.target_network(next_states).max(1)[0]
            target_q   = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Décroissance epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Mise à jour réseau cible
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
