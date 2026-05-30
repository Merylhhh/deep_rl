# 🤖 Deep Reinforcement Learning — CartPole-v1

> Implémentation from scratch de 3 algorithmes Deep RL avec PyTorch  
> Environnement : **CartPole-v1** (OpenAI Gymnasium)

---

## 📋 Algorithmes implémentés

| Algorithme | Famille | Fichier | Principe |
|---|---|---|---|
| **DQN** | Value-Based | `1_DQN/` | Apprendre Q(s,a) avec un réseau neuronal + Replay Buffer |
| **REINFORCE** | Policy Gradient | `2_PolicyGradient/` | Optimiser π(a\|s) directement par gradient de politique |
| **A2C** | Actor-Critic | `3_ActorCritic/` | Combiner acteur (politique) et critique (valeur) |

---

## 🧠 Concepts clés

### 1. Deep Q-Network (DQN)
Le DQN apprend une fonction de valeur Q(s,a) approchée par un réseau neuronal.  
Innovations clés : **Experience Replay** (mémoire de transitions) + **Target Network** (réseau cible figé pour stabiliser l'apprentissage).

```
Q(s,a) ← r + γ · max_a' Q_target(s', a')
```

### 2. Policy Gradient (REINFORCE)
Méthode Monte-Carlo qui optimise directement la politique stochastique π_θ.  
L'agent collecte un épisode complet, calcule les retours discountés G_t, puis remonte le gradient.

```
∇J(θ) = E[∇ log π_θ(a|s) · G_t]
```

### 3. Advantage Actor-Critic (A2C)
Architecture partagée avec deux têtes :
- **Acteur** : π_θ(a|s) → choisit les actions
- **Critique** : V_θ(s) → évalue les états

L'avantage A(s,a) = r + γV(s') - V(s) réduit la variance par rapport à REINFORCE.

```
L_total = L_acteur + α·L_critique - β·H(π)
```

---

## 🗂️ Structure du projet

```
deep-rl-algorithms/
├── README.md
├── requirements.txt
├── compare_all.py          ← Lance et compare les 3 algos
│
├── 1_DQN/
│   ├── dqn.py              ← QNetwork, ReplayBuffer, DQNAgent
│   └── train.py            ← Entraînement + courbe + GIF
│
├── 2_PolicyGradient/
│   ├── policy_network.py   ← PolicyNetwork, REINFORCEAgent
│   └── train.py
│
├── 3_ActorCritic/
│   ├── actor_critic.py     ← ActorCriticNetwork, A2CAgent
│   └── train.py
│
└── results/
    ├── plots/              ← Courbes d'apprentissage (.png)
    └── gifs/               ← Animations de l'agent (.gif)
```

---

## 🚀 Installation & Utilisation

```bash
# 1. Cloner le repo
git clone https://github.com/<votre-username>/deep-rl-algorithms.git
cd deep-rl-algorithms

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner un algorithme
cd 1_DQN    && python train.py
cd 2_PolicyGradient && python train.py
cd 3_ActorCritic    && python train.py

# 4. Comparer les 3 algorithmes
python compare_all.py
```

---

## 📊 Résultats

Les courbes d'apprentissage et animations GIF sont générées automatiquement dans `results/`.

CartPole-v1 est considéré **résolu** lorsque la moyenne des récompenses sur 100 épisodes consécutifs atteint **≥ 475**.

---

## 🔧 Hyperparamètres

### DQN
| Paramètre | Valeur |
|---|---|
| Learning rate | 1e-3 |
| Discount γ | 0.99 |
| ε initial / min | 1.0 / 0.01 |
| Décroissance ε | 0.995 |
| Batch size | 64 |
| Replay buffer | 10 000 |
| Target update | tous les 10 pas |

### REINFORCE
| Paramètre | Valeur |
|---|---|
| Learning rate | 1e-3 |
| Discount γ | 0.99 |

### A2C
| Paramètre | Valeur |
|---|---|
| Learning rate | 3e-4 |
| Discount γ | 0.99 |
| Coeff. critique | 0.5 |
| Coeff. entropie | 0.01 |
| Gradient clipping | 0.5 |

---

## 📚 Références

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Williams (1992) — *Simple statistical gradient-following algorithms* (REINFORCE)
- Mnih et al. (2016) — *Asynchronous methods for deep reinforcement learning* (A3C/A2C)
- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)

---

*Cours Deep Reinforcement Learning — 2025/2026*
