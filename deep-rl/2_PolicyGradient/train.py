import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_network import REINFORCEAgent

def train(n_episodes=800, max_steps=500, render_last=True):
    env = gym.make("CartPole-v1")
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size)

    rewards_history = []
    avg_rewards     = []
    solved_episode  = None

    print("=" * 55)
    print("   REINFORCE — Policy Gradient — CartPole-v1")
    print("=" * 55)

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            state        = next_state
            total_reward += reward

            if done:
                break

        # Mise à jour politique (fin d'épisode)
        agent.train_step()

        rewards_history.append(total_reward)
        avg = np.mean(rewards_history[-100:])
        avg_rewards.append(avg)

        if ep % 50 == 0:
            print(f"  Épisode {ep:4d} | Récompense : {total_reward:6.1f} | Moy-100 : {avg:6.1f}")

        if avg >= 475 and solved_episode is None:
            solved_episode = ep
            print(f"\n   Résolu à l'épisode {ep} !\n")

    env.close()

    os.makedirs("../results/plots", exist_ok=True)
    _plot(rewards_history, avg_rewards, solved_episode, algo="PolicyGradient")

    with open("../results/pg_rewards.pkl", "wb") as f:
        pickle.dump(rewards_history, f)

    if render_last:
        _record_gif(agent, filename="../results/gifs/pg_cartpole.gif")

    return agent, rewards_history


def _plot(rewards, avgs, solved_ep, algo="PolicyGradient"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.4, color="#CE93D8", linewidth=0.8, label="Récompense par épisode")
    ax.plot(avgs,    color="#7B1FA2", linewidth=2,   label="Moyenne mobile (100 épisodes)")
    ax.axhline(475, color="#EF5350", linestyle="--", linewidth=1.2, label="Seuil de résolution (475)")
    if solved_ep:
        ax.axvline(solved_ep, color="#66BB6A", linestyle=":", linewidth=1.5,
                   label=f"Résolu — épisode {solved_ep}")
    ax.set_xlabel("Épisode",    fontsize=12)
    ax.set_ylabel("Récompense", fontsize=12)
    ax.set_title(f"{algo} — Courbe d'apprentissage CartPole-v1", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"../results/plots/{algo.lower()}_learning_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Courbe sauvegardée → {path}")


def _record_gif(agent, filename="../results/gifs/pg_cartpole.gif", n_steps=300):
    try:
        import imageio
        env    = gym.make("CartPole-v1", render_mode="rgb_array")
        frames = []
        state, _ = env.reset()

        for _ in range(n_steps):
            frames.append(env.render())
            action = agent.select_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()
        imageio.mimsave(filename, frames, fps=30)
        print(f"   GIF sauvegardé → {filename}")
    except ImportError:
        print("    imageio non installé — GIF ignoré")


if __name__ == "__main__":
    train()
