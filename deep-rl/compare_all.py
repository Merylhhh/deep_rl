"""
compare_all.py
──────────────
Lance les 3 algorithmes et génère un graphique comparatif.
Usage : python compare_all.py
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Ajout des modules au path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "1_DQN"))
sys.path.insert(0, os.path.join(BASE, "2_PolicyGradient"))
sys.path.insert(0, os.path.join(BASE, "3_ActorCritic"))

os.chdir(BASE)

from dqn            import DQNAgent
from policy_network import REINFORCEAgent
from actor_critic   import A2CAgent
import gymnasium as gym


def run_agent(AgentClass, agent_kwargs, n_episodes=500, max_steps=500, label=""):
    env = gym.make("CartPole-v1")
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = AgentClass(state_size, action_size, **agent_kwargs)
    rewards = []
    print(f"\n  ▶ Entraînement {label}...")

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total = 0
        for _ in range(max_steps):
            # Sélection d'action selon le type d'agent
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state)
                ns, r, term, trunc, _ = env.step(action)
                done = term or trunc
                agent.store(state, action, r, ns, float(done))
                agent.train_step()
            else:
                action = agent.select_action(state)
                ns, r, term, trunc, _ = env.step(action)
                done = term or trunc
                agent.store_reward(r)
            state  = ns
            total += r
            if done:
                break

        if not isinstance(agent, DQNAgent):
            kw = {"last_state": state, "done": done} if isinstance(agent, A2CAgent) else {}
            agent.train_step(**kw)

        rewards.append(total)
        if ep % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"    Épisode {ep:4d} | Moy-100 : {avg:.1f}")

    env.close()
    return rewards


def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode='valid')


if __name__ == "__main__":
    N = 500

    dqn_r = run_agent(DQNAgent,       {},              n_episodes=N, label="DQN")
    pg_r  = run_agent(REINFORCEAgent, {},              n_episodes=N, label="REINFORCE")
    ac_r  = run_agent(A2CAgent,       {},              n_episodes=N, label="A2C")

    os.makedirs("results/plots", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Comparaison DQN vs REINFORCE vs A2C — CartPole-v1",
                 fontsize=15, fontweight="bold")

    colors = {"DQN": "#0288D1", "REINFORCE": "#7B1FA2", "A2C": "#2E7D32"}

    # ── Courbes brutes ──
    ax = axes[0]
    for label, rewards, c in [("DQN", dqn_r, colors["DQN"]),
                               ("REINFORCE", pg_r, colors["REINFORCE"]),
                               ("A2C", ac_r, colors["A2C"])]:
        ax.plot(rewards, alpha=0.3, color=c, linewidth=0.7)
        ax.plot(smooth(rewards), color=c, linewidth=2, label=label)
    ax.axhline(475, color="#EF5350", linestyle="--", linewidth=1.2, label="Seuil (475)")
    ax.set_xlabel("Épisode"); ax.set_ylabel("Récompense")
    ax.set_title("Courbes d'apprentissage lissées"); ax.legend(); ax.grid(alpha=0.3)

    # ── Box plot des 100 derniers épisodes ──
    ax2 = axes[1]
    data = [dqn_r[-100:], pg_r[-100:], ac_r[-100:]]
    bp   = ax2.boxplot(data, patch_artist=True, labels=["DQN", "REINFORCE", "A2C"])
    for patch, c in zip(bp["boxes"], list(colors.values())):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax2.axhline(475, color="#EF5350", linestyle="--", linewidth=1.2, label="Seuil (475)")
    ax2.set_ylabel("Récompense (100 derniers épisodes)")
    ax2.set_title("Distribution des performances finales")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = "results/plots/comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Comparaison sauvegardée → {path}")

    # Résumé
    print("\n" + "=" * 45)
    print("  RÉSUMÉ — Moyenne des 100 derniers épisodes")
    print("=" * 45)
    for label, r in [("DQN", dqn_r), ("REINFORCE", pg_r), ("A2C", ac_r)]:
        print(f"  {label:<12} : {np.mean(r[-100:]):6.1f} ± {np.std(r[-100:]):.1f}")
