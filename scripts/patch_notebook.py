import json

with open('notebooks/dqn_spot_demo.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

new_source = '''from tqdm.notebook import tqdm

def train_dqn(env, agent, num_episodes=1500):
    rewards_history = []
    loss_history    = []
    epsilon_history = []
    cost_history    = []
    sla_history     = []
    best_avg_reward = -float("inf")

    pbar = tqdm(range(num_episodes), desc="Training", unit="ep")

    for episode in pbar:
        obs, info = env.reset()
        episode_reward = 0.0
        episode_losses = []

        done = False
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            episode_reward += reward
            obs = next_obs

        rewards_history.append(episode_reward)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)
        epsilon_history.append(agent.epsilon)
        cost_history.append(info.get("cost", 0.0))
        sla_history.append(info.get("sla_compliance", 1.0))

        avg_r = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
        if len(rewards_history) >= 100 and avg_r > best_avg_reward:
            best_avg_reward = avg_r
            agent.save("../results/notebook_demo/best_model.pth")

        if (episode + 1) % 10 == 0:
            avg_loss = np.mean(loss_history[-10:])
            avg_sla  = np.mean(sla_history[-10:])
            pbar.set_postfix({
                "reward": f"{avg_r:.1f}",
                "loss":   f"{avg_loss:.3f}",
                "eps":    f"{agent.epsilon:.3f}",
                "sla":    f"{avg_sla:.2%}",
                "cost":   f"${np.mean(cost_history[-10:]):.1f}",
            })

    pbar.close()
    print(f"\\nTraining complete!")
    print(f"Best avg reward (100ep): {best_avg_reward:.2f}")
    return rewards_history, loss_history, epsilon_history, cost_history, sla_history


import os
os.makedirs("../results/notebook_demo", exist_ok=True)

print(f"Config: spot_capacity={SPOT_CAPACITY}, ondemand_capacity={ONDEMAND_CAPACITY}, SCALE_STEP={env.SCALE_STEP}")
rewards_hist, loss_hist, eps_hist, cost_hist, sla_hist = train_dqn(env, agent, num_episodes=1500)
'''

nb['cells'][9]['source'] = [new_source]

with open('notebooks/dqn_spot_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Done! Cell 9 updated with tqdm progress bar.')
