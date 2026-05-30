import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ─────────────────────────────────────────
#  Réseau de politique (Policy Network)
# ─────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """
    Réseau stochastique : prend un état en entrée et produit
    une distribution de probabilité sur les actions.
    """
    def __init__(self, state_size, action_size, hidden=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
            nn.Softmax(dim=-1)           # probabilités d'actions
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
#  Agent REINFORCE (Policy Gradient)
# ─────────────────────────────────────────
class REINFORCEAgent:
    """
    Algorithme REINFORCE (Monte-Carlo Policy Gradient).
    Met à jour la politique en fin d'épisode avec les retours
    cumulés discountés G_t.
    """
    def __init__(self, state_size, action_size,
                 lr=1e-3, gamma=0.99):

        self.gamma  = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy    = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Mémoire d'épisode
        self.log_probs = []
        self.rewards   = []

    def select_action(self, state):
        """Échantillonne une action selon π_θ(a|s)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs   = self.policy(state_t)
        dist    = torch.distributions.Categorical(probs)
        action  = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def train_step(self):
        """Mise à jour de la politique en fin d'épisode (Monte-Carlo)."""
        # Calcul des retours discountés G_t
        G, returns = 0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalisation (stabilité d'entraînement)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Perte REINFORCE : -∑ log π(a|s) · G_t
        loss = torch.stack(
            [-lp * G_t for lp, G_t in zip(self.log_probs, returns)]
        ).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Réinitialisation de la mémoire d'épisode
        self.log_probs = []
        self.rewards   = []

        return loss.item()
