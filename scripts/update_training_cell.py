"""Rebuild cell 8 for from-scratch train: no greedy-eval drift, robust early stop."""
import json

NB = 'notebooks/kaggle_train.ipynb'

training_src = r'''# Step 6: Training -- from-scratch generalist, prioritized scenario sampling + MLflow
import time, pickle, os, mlflow
import numpy as np
from collections import defaultdict

if 'MLFLOW_URI' not in dir():
    MLFLOW_URI = None

NUM_EPISODES = 8000
MIN_EPISODES = 4000
PATIENCE     = 1200     # episodes without improvement (large, permissive)
EPS_GATE     = 0.08     # only early-stop after epsilon decayed below this
PRINT_EVERY  = 100
LOG_EVERY    = 10
WINDOW       = 300      # moving-average window for scoring
os.makedirs(SAVE_DIR, exist_ok=True)

def model_score(costs, slas, window=WINDOW):
    """Lexicographic: SLA >= 95% gate, then minimise cost."""
    if len(costs) < window:
        return -float('inf')
    recent_cost = np.mean(costs[-window:])
    recent_sla  = np.mean(slas[-window:])
    if recent_sla < 0.95:
        return -1e6 + recent_sla * 1e5
    return -recent_cost

rewards_history, loss_history = [], []
cost_history, sla_history, eps_history = [], [], []
scenario_history = []
per_scn_cost = defaultdict(list)
per_scn_sla  = defaultdict(list)

best_score  = -float('inf')
best_cost   = best_sla = best_reward = None
no_improve_count = 0
start_time = time.time()

run_name = f'resume-ep{agent.steps_done//168}' if RESUME_CHECKPOINT else 'generalist-scratch-8k'
run_ctx  = mlflow.start_run(run_name=run_name) if MLFLOW_URI else None
if run_ctx:
    mlflow.log_params({
        **{k: v for k, v in agent_cfg.items() if not isinstance(v, (list, dict))},
        'num_episodes':  NUM_EPISODES,
        'min_episodes':  MIN_EPISODES,
        'patience':      PATIENCE,
        'eps_gate':      EPS_GATE,
        'score_window':  WINDOW,
        'scenarios':     ','.join(SCENARIOS.keys()),
        'network_params': sum(p.numel() for p in agent.q_network.parameters()),
        'prioritized_scenario_sampling': True,
        'resumed_from':  RESUME_CHECKPOINT or 'scratch',
        'resume_epsilon': agent.epsilon,
        'resume_steps':   agent.steps_done,
    })

try:
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        scn = info['scenario']
        ep_reward, ep_losses = 0.0, []

        for step in range(168):
            mask = env.get_action_mask()
            action = agent.select_action(obs, training=True, action_mask=mask)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            ep_reward += reward
            obs = next_obs
            if done:
                break

        ep_cost = info.get('cost', 0)
        ep_sla  = info.get('sla_compliance', 0)
        ep_loss = np.mean(ep_losses) if ep_losses else 0.0

        rewards_history.append(ep_reward)
        loss_history.append(ep_loss)
        eps_history.append(agent.epsilon)
        cost_history.append(ep_cost)
        sla_history.append(ep_sla)
        scenario_history.append(scn)
        per_scn_cost[scn].append(ep_cost)
        per_scn_sla[scn].append(ep_sla)

        if ep_loss > 0:
            env.update_scenario_loss(scn, ep_loss)

        # Save best only after warmup AND once moving average is stable
        if episode + 1 >= max(WINDOW, MIN_EPISODES // 2):
            score = model_score(cost_history, sla_history, window=WINDOW)
            if score > best_score + 1.0:
                best_score  = score
                best_cost   = float(np.mean(cost_history[-WINDOW:]))
                best_sla    = float(np.mean(sla_history[-WINDOW:]))
                best_reward = float(np.mean(rewards_history[-WINDOW:]))
                no_improve_count = 0
                agent.save(f'{SAVE_DIR}/best_mixed.pth')
                if run_ctx:
                    mlflow.log_metrics({
                        'best_cost': best_cost,
                        'best_sla':  best_sla,
                        'best_reward': best_reward,
                    }, step=episode)
                    try:
                        mlflow.log_artifact(f'{SAVE_DIR}/best_mixed.pth', artifact_path='models')
                    except Exception as _e:
                        print(f'Artifact upload skipped: {_e}')
            elif episode >= MIN_EPISODES:
                no_improve_count += 1

        if run_ctx and (episode + 1) % LOG_EVERY == 0:
            metrics = {
                'reward':  float(np.mean(rewards_history[-LOG_EVERY:])),
                'cost':    float(np.mean(cost_history[-LOG_EVERY:])),
                'sla':     float(np.mean(sla_history[-LOG_EVERY:])),
                'loss':    float(np.mean(loss_history[-LOG_EVERY:])),
                'epsilon': float(agent.epsilon),
            }
            for n in ['stable', 'volatile', 'spike', 'az_divergence']:
                if per_scn_cost[n]:
                    metrics[f'cost_{n}'] = float(np.mean(per_scn_cost[n][-20:]))
                    metrics[f'sla_{n}']  = float(np.mean(per_scn_sla[n][-20:]))
            for i, n in enumerate(env.scenario_names):
                metrics[f'weight_{n}'] = float(env.weights[i])
            mlflow.log_metrics(metrics, step=episode)

        if (episode + 1) % PRINT_EVERY == 0:
            avg_r = np.mean(rewards_history[-100:])
            avg_c = np.mean(cost_history[-100:])
            avg_s = np.mean(sla_history[-100:])
            elapsed = (time.time() - start_time) / 60
            bc = f'${best_cost:.1f}' if best_cost is not None else 'n/a'
            bs = f'{best_sla:.1%}'  if best_sla  is not None else 'n/a'
            scn_sum = ' '.join(
                f'{n[:3]}:${np.mean(per_scn_cost[n][-50:]):.0f}/{np.mean(per_scn_sla[n][-50:]):.0%}'
                for n in ['stable','volatile','spike','az_divergence'] if per_scn_cost[n]
            )
            w_str = ' '.join(f'{n[:3]}:{env.weights[i]:.2f}' for i, n in enumerate(env.scenario_names))
            print(f'Ep {episode+1:4d} | R:{avg_r:+7.1f} Cost:${avg_c:5.1f} SLA:{avg_s:.1%} '
                  f'Eps:{agent.epsilon:.3f} best:{bc}/{bs} | {scn_sum} | w:[{w_str}] | {elapsed:.1f}m')

        # Early stop: epsilon-gated + patience-gated
        if (episode >= MIN_EPISODES
                and agent.epsilon < EPS_GATE
                and no_improve_count >= PATIENCE):
            print(f'Early stop ep {episode+1}. '
                  f'Best cost=${best_cost:.1f} SLA={best_sla:.1%} eps={agent.epsilon:.3f}')
            break

finally:
    agent.save(f'{SAVE_DIR}/final_mixed.pth')
    with open(f'{SAVE_DIR}/history_mixed.pkl', 'wb') as f:
        pickle.dump({
            'rewards': rewards_history, 'loss': loss_history,
            'epsilon': eps_history, 'cost': cost_history, 'sla': sla_history,
            'scenario': scenario_history,
            'per_scn_cost': dict(per_scn_cost), 'per_scn_sla': dict(per_scn_sla),
        }, f)
    if run_ctx:
        try:
            mlflow.log_artifact(f'{SAVE_DIR}/final_mixed.pth',   artifact_path='models')
            mlflow.log_artifact(f'{SAVE_DIR}/history_mixed.pkl', artifact_path='history')
        except Exception as _e:
            print(f'Final artifact upload skipped: {_e}')
        mlflow.end_run()
        print('MLflow run ended.')

# Register best model to MLflow Model Registry
if run_ctx and best_cost is not None:
    try:
        run_id    = run_ctx.info.run_id
        model_uri = f'runs:/{run_id}/models/best_mixed.pth'
        mv = mlflow.register_model(model_uri=model_uri, name='Spot-RL-Agent')
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(mv.name, mv.version, 'avg_cost',    f'{best_cost:.1f}')
        client.set_model_version_tag(mv.name, mv.version, 'avg_sla',     f'{best_sla:.3f}')
        client.set_model_version_tag(mv.name, mv.version, 'epsilon_end', f'{agent.epsilon:.4f}')
        client.set_model_version_tag(mv.name, mv.version, 'steps_done',  str(agent.steps_done))
        client.set_model_version_tag(mv.name, mv.version, 'resumed_from', RESUME_CHECKPOINT or 'scratch')
        print(f'Registered: Spot-RL-Agent v{mv.version} | cost=${best_cost:.1f} SLA={best_sla:.1%}')
    except Exception as _e:
        print(f'Model Registry skipped: {_e}')

print(f'Done. Best: cost=${best_cost:.1f} SLA={best_sla:.1%} R={best_reward:+.1f}')
'''

nb = json.load(open(NB, encoding='utf-8'))
nb['cells'][8]['source'] = training_src
nb['cells'][8]['outputs'] = []
nb['cells'][8]['execution_count'] = None
json.dump(nb, open(NB, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Updated cell 8 in {NB}')
print('  NUM_EPISODES=8000, MIN_EPISODES=4000, PATIENCE=1200, EPS_GATE=0.08, WINDOW=300')
print('  Removed greedy_eval (avoids scenario-weight drift).')
