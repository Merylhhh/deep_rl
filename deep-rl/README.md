#  Deep Reinforcement Learning — CartPole-v1

> Implémentation from scratch de 3 algorithmes Deep RL avec PyTorch  
> Environnement : **CartPole-v1** (OpenAI Gymnasium)

---

##  Algorithmes implémentés

| Algorithme | Famille | Fichier | Idée centrale |
|---|---|---|---|
| **DQN** | Value-Based | `1_DQN/` | Apprendre *combien vaut* chaque action |
| **REINFORCE** | Policy Gradient | `2_PolicyGradient/` | Apprendre *directement* quelle action choisir |
| **A2C** | Actor-Critic | `3_ActorCritic/` | Combiner les deux approches ci-dessus |

---

##  L'environnement : CartPole-v1

Avant de parler des algorithmes, comprenons le problème qu'ils doivent résoudre.

CartPole est un chariot sur un rail avec une perche posée dessus. L'agent doit **empêcher la perche de tomber** en poussant le chariot à gauche ou à droite.

- **État (observation)** : 4 valeurs — position du chariot, sa vitesse, angle de la perche, vitesse angulaire
- **Actions** : 2 choix — pousser à gauche (0) ou à droite (1)
- **Récompense** : +1 à chaque instant où la perche reste debout
- **Fin d'épisode** : la perche tombe (angle > 15°) ou le chariot sort des rails
- **Résolu** : moyenne ≥ 475 points sur 100 épisodes consécutifs

L'agent commence sans rien savoir. Il apprend uniquement par **essais et erreurs**, en observant les conséquences de ses actions.

---

##  Les 3 algorithmes expliqués simplement

---

### 1️ Deep Q-Network (DQN)

#### L'intuition

Imagine que tu joues à un jeu vidéo pour la première fois. Naturellement, tu mémorises des situations : *"quand la perche penche à droite, pousser à droite m'a donné beaucoup de points"*. Avec le temps, tu construis une **carte mentale** qui associe chaque situation à la valeur de chaque action possible.

C'est exactement ce que fait DQN : il apprend une fonction **Q(état, action)** qui répond à la question *"si je suis dans cet état et que je fais cette action, combien de récompense vais-je obtenir en tout ?"*

#### Comment ça marche en pratique

**1. Le réseau de neurones remplace le tableau**  
En théorie, on pourrait stocker Q(s,a) dans un tableau. Mais les états sont des nombres réels (infinis possibilités), donc on utilise un **réseau neuronal** qui *généralise* : il apprend à estimer Q pour des états qu'il n'a jamais vus exactement.

**2. L'Experience Replay (mémoire)**  
Sans cette astuce, l'agent apprendrait séquentiellement (épisode 1, puis 2, puis 3...) ce qui crée des corrélations et déstabilise l'apprentissage. Avec le Replay Buffer, chaque transition `(état, action, récompense, état suivant)` est stockée en mémoire. À chaque étape, on tire un **mini-batch aléatoire** de 64 expériences passées pour s'entraîner. C'est comme réviser ses erreurs passées dans le désordre.

**3. Le Target Network (réseau cible)**  
Un problème subtil : si on utilise le même réseau pour calculer la cible ET pour s'entraîner, la cible bouge à chaque mise à jour → instabilité. La solution : on garde une **copie figée** du réseau (le target network) pour calculer les cibles, et on la synchronise avec le réseau principal toutes les 10 étapes seulement.

**4. La politique ε-greedy**  
Au début, l'agent explore aléatoirement (ε = 1.0 = 100% aléatoire). Progressivement, ε diminue : l'agent exploite de plus en plus ce qu'il a appris, tout en gardant une petite part d'exploration (ε_min = 0.01).

#### Les étapes d'un épisode DQN

```
Pour chaque épisode :
  Observer l'état s
  Choisir action a (aléatoire avec proba ε, sinon argmax Q(s,·))
  Exécuter a → obtenir récompense r et nouvel état s'
  Stocker (s, a, r, s', done) dans le Replay Buffer
  Tirer 64 expériences aléatoires du buffer
  Calculer la cible : y = r + γ · max Q_target(s', ·)
  Minimiser (Q(s,a) - y)²  par descente de gradient
  Mettre à jour ε
```

#### Points forts et limites

 Très stable grâce au replay buffer et au target network  
 Efficace en termes de données (réutilise les expériences passées)  
 Ne fonctionne qu'avec des **actions discrètes** (pas de contrôle continu)  
 Peut sous-estimer ou surestimer les valeurs Q (biais d'optimisme)

---

### 2️ Policy Gradient — REINFORCE

#### L'intuition

Au lieu d'apprendre *combien vaut* chaque action, pourquoi ne pas apprendre **directement** une politique — c'est-à-dire une règle qui dit *"dans cette situation, fais ça"* ?

REINFORCE fait exactement ça. La politique est un réseau neuronal qui prend un état en entrée et sort une **distribution de probabilités** sur les actions. L'idée centrale : après un épisode, **augmenter la probabilité des actions qui ont mené à de bonnes récompenses, et diminuer celles des mauvaises actions**.

C'est comme un coach sportif : après chaque match, il dit *"le tir que tu as fait à la 30e minute était excellent, refais-le"* ou *"cette passe à la 60e était mauvaise, évite-la"*.

#### Comment ça marche en pratique

**1. La politique stochastique**  
Contrairement à DQN qui choisit toujours l'action avec le plus grand Q (déterministe), REINFORCE *tire au sort* une action selon les probabilités. Si la politique dit `[0.8, 0.2]`, il y a 80% de chance de choisir action 0. Cette stochasticité assure l'exploration naturellement, sans besoin d'ε-greedy.

**2. Les retours discountés G_t**  
À la fin de l'épisode, on calcule pour chaque instant t le **retour cumulé** :  
`G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...`  
γ = 0.99 signifie que les récompenses futures comptent presque autant que les immédiates, mais légèrement moins (on préfère les récompenses maintenant).

**3. La mise à jour**  
On augmente la log-probabilité des actions proportionnellement à leur retour G_t. Si une action a mené à G_t = 450, on l'encourage fortement. Si G_t = 10, on l'encourage peu (ou la décourage si normalisé).

**4. Monte-Carlo = épisode complet**  
REINFORCE attend la **fin de l'épisode entier** pour apprendre. C'est son principal défaut : si un épisode dure 500 étapes, on attend 500 étapes avant la moindre mise à jour.

#### Les étapes d'un épisode REINFORCE

```
Pour chaque épisode :
  Jouer l'épisode COMPLET en suivant π_θ
  Collecter toutes les (action, log_prob, récompense)
  Calculer G_t pour chaque étape (de la fin vers le début)
  Normaliser les G_t (moyenne 0, écart-type 1)
  Mettre à jour θ : maximiser Σ log π_θ(a_t|s_t) · G_t
```

#### Points forts et limites

 Fonctionne avec des espaces d'actions **continus** (robotique, etc.)  
 Directement optimise ce qu'on veut (la politique)  
 **Haute variance** : deux épisodes similaires peuvent donner des mises à jour très différentes  
 Lent à converger (besoin de nombreux épisodes)  
 Doit finir l'épisode avant d'apprendre

---

### 3️ Advantage Actor-Critic (A2C)

#### L'intuition

A2C est la synthèse intelligente des deux approches précédentes. Il garde deux têtes :

- **L'Acteur** (comme REINFORCE) : choisit les actions, optimise la politique directement
- **Le Critique** (comme DQN) : évalue la qualité d'un état, estime V(s)

La grande innovation : au lieu d'utiliser le retour brut G_t (très bruité), l'acteur utilise l'**avantage** :

```
A(s, a) = r + γ·V(s') - V(s)
```

L'avantage répond à : *"cette action était-elle meilleure ou moins bonne que ce que j'attendais normalement ?"*  
Si A > 0 → l'action était meilleure qu'attendu → on l'encourage  
Si A < 0 → l'action était pire qu'attendu → on la décourage

#### La métaphore du stagiaire et du manager

- Le **Critique** est le manager expérimenté qui sait évaluer une situation : *"en ce moment, on est en bonne position, je m'attends à 300 points"*
- L'**Acteur** est le stagiaire qui agit. Il compare son résultat aux attentes du manager : si il obtient 400 alors que le manager attendait 300, l'avantage est +100 → excellente action !

Le critique apprend en continu, ce qui donne à l'acteur un **signal de comparaison stable** — bien moins bruité que G_t pur.

#### Architecture partagée

Les couches de base (extraction de features) sont **partagées** entre acteur et critique. Seules les dernières couches divergent. Cela permet de :
- Économiser des paramètres
- Partager la représentation de l'environnement
- Accélérer l'apprentissage

#### Les 3 composantes de la perte

```
L_total = L_acteur  +  0.5 · L_critique  -  0.01 · Entropie

L_acteur   = -log π(a|s) · Avantage        (maximiser les bonnes actions)
L_critique = (V(s) - G_t)²                 (améliorer les prédictions du critique)
-Entropie                                  (encourager l'exploration)
```

Le terme d'entropie est crucial : il empêche la politique de devenir trop déterministe trop tôt, forçant l'agent à explorer.

#### Points forts et limites

 **Moins de variance** que REINFORCE grâce à l'avantage  
 Apprend **à chaque étape** (pas besoin de finir l'épisode)  
 Fonctionne avec actions discrètes ET continues  
 Généralement le plus stable des 3 ici  
 Plus complexe à implémenter et déboguer  
 Sensible au choix des hyperparamètres (notamment les coefficients de perte)

---

##  Comparaison des 3 algorithmes

| Critère | DQN | REINFORCE | A2C |
|---|---|---|---|
| **Ce qu'il apprend** | Valeur Q(s,a) | Politique π(a\|s) | Les deux |
| **Quand il apprend** | À chaque pas | Fin d'épisode | À chaque pas |
| **Variance** | Faible | Très haute | Moyenne |
| **Stabilité** | ✅✅ | ⚠️ | ✅✅✅ |
| **Vitesse de convergence** | Rapide | Lent | Rapide |
| **Mémoire requise** | Élevée (buffer) | Faible | Faible |
| **Actions continues** | ❌ | ✅ | ✅ |
| **Complexité** | Moyenne | Simple | Élevée |

---

##  Structure du projet

```
deep-rl-algorithms/
├── README.md
├── requirements.txt
├── compare_all.py              ← Lance et compare les 3 algos
│
├── 1_DQN/
│   ├── dqn.py                  ← QNetwork, ReplayBuffer, DQNAgent
│   └── train.py                ← Entraînement + courbe + GIF
│
├── 2_PolicyGradient/
│   ├── policy_network.py       ← PolicyNetwork, REINFORCEAgent
│   └── train.py
│
├── 3_ActorCritic/
│   ├── actor_critic.py         ← ActorCriticNetwork (têtes partagées) + A2CAgent
│   └── train.py
│
└── results/
    ├── plots/                  ← Courbes d'apprentissage (.png)
    └── gifs/                   ← Animations de l'agent (.gif)
```

---

##  Installation & Utilisation

```bash
# 1. Cloner le repo
git clone https://github.com/<votre-username>/deep-rl-algorithms.git
cd deep-rl-algorithms

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner un algorithme individuel
cd 1_DQN            && python train.py
cd ../2_PolicyGradient   && python train.py
cd ../3_ActorCritic      && python train.py

# 4. Comparer les 3 algorithmes
cd ..
python compare_all.py
```

Les courbes PNG et GIFs sont générés automatiquement dans `results/`.

---

##  Résultats

CartPole-v1 est considéré **résolu** lorsque la moyenne des récompenses sur 100 épisodes consécutifs atteint **≥ 475** points.

---

##  Hyperparamètres

### DQN
| Paramètre | Valeur | Rôle |
|---|---|---|
| Learning rate | 1e-3 | Vitesse d'apprentissage du réseau |
| Discount γ | 0.99 | Importance des récompenses futures |
| ε initial / min | 1.0 / 0.01 | Taux d'exploration initial et minimal |
| Décroissance ε | 0.995 | Vitesse de passage exploration → exploitation |
| Batch size | 64 | Nombre d'expériences par mise à jour |
| Replay buffer | 10 000 | Taille de la mémoire d'expériences |
| Target update | tous les 10 pas | Fréquence de synchronisation du réseau cible |

### REINFORCE
| Paramètre | Valeur | Rôle |
|---|---|---|
| Learning rate | 1e-3 | Vitesse d'apprentissage de la politique |
| Discount γ | 0.99 | Importance des récompenses futures |

### A2C
| Paramètre | Valeur | Rôle |
|---|---|---|
| Learning rate | 3e-4 | Plus faible pour plus de stabilité |
| Discount γ | 0.99 | Importance des récompenses futures |
| Coeff. critique | 0.5 | Poids de la perte du critique |
| Coeff. entropie | 0.01 | Force l'exploration |
| Gradient clipping | 0.5 | Évite les mises à jour explosives |

---

##  Références

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Williams (1992) — *Simple statistical gradient-following algorithms* (REINFORCE)
- Mnih et al. (2016) — *Asynchronous methods for deep reinforcement learning* (A3C/A2C)
- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)

---

