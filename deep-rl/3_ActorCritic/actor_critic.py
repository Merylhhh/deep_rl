import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ─────────────────────────────────────────
#  Réseau Acteur-Critique partagé
# ─────────────────────────────────────────
class ActorCriticNetwork(nn.Module):
    """
    Architecture partagée avec deux têtes :
      - Acteur  : π_θ(a|s)  → distribution d'actions
      - Critique: V_θ(s)    → valeur d'état
    """
    def __init__(self, state_size, action_size, hidden=128):
        super(ActorCriticNetwork, self).__init__()

        # Couches partagées (feature extraction)
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Tête Acteur
        self.actor_head  = nn.Sequential(
            nn.Linear(hidden, action_size),
            nn.Softmax(dim=-1)
        )

        # Tête Critique
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        probs    = self.actor_head(features)   # π(a|s)
        value    = self.critic_head(features)  # V(s)
        return probs, value


# ─────────────────────────────────────────
#  Agent A2C (Advantage Actor-Critic)
# ─────────────────────────────────────────
class A2CAgent:
    """
    Advantage Actor-Critic (A2C) — variante synchrone de A3C.
    Avantage : A(s,a) = r + γV(s') - V(s)  (réduit la variance)
    """
    def __init__(self, state_size, action_size,
                 lr=3e-4, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01):

        self.gamma        = gamma
        self.value_coef   = value_coef    # pondération perte critique
        self.entropy_coef = entropy_coef  # régularisation entropie
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network   = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Mémoire d'épisode
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.entropies = []

    def select_action(self, state):
        """Sélectionne une action et stocke log-prob, valeur, entropie."""
        state_t       = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value  = self.network(state_t)

        dist    = torch.distributions.Categorical(probs)
        action  = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.values.append(value.squeeze())
        self.entropies.append(dist.entropy())

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def train_step(self, last_state=None, done=True):
        """
        Mise à jour Actor-Critic.
        Perte totale = L_acteur + value_coef·L_critique - entropy_coef·H
        """
        # Retours discountés
        R = 0 if done else self.network(
            torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        )[1].item()

        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns    = torch.FloatTensor(returns).to(self.device)
        values_t   = torch.stack(self.values)
        log_probs_t = torch.stack(self.log_probs)
        entropies_t = torch.stack(self.entropies)

        # Avantage A(s,a) = G_t - V(s)
        advantages = returns - values_t.detach()

        # Normalisation de l'avantage
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Pertes
        actor_loss  = -(log_probs_t * advantages).mean()
        critic_loss = nn.functional.mse_loss(values_t, returns)
        entropy     = entropies_t.mean()

        total_loss  = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Réinitialisation
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.entropies = []

        return total_loss.item(), actor_loss.item(), critic_loss.item()
