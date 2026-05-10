// CI/CD pipeline — chạy trên Spot agent (label: spot-worker)
// Stages: setup → train (short) → export ONNX → archive
pipeline {
    agent { label 'spot-worker' }

    options {
        timeout(time: 30, unit: 'MINUTES')
        retry(1)
        buildDiscarder(logRotator(numToKeepStr: '5'))
    }

    environment {
        PYTHON      = 'python3'
        RESULTS_DIR = 'results/kaggle'
        MODEL_OUT   = "results/kaggle/jenkins_build_${BUILD_NUMBER}.pth"
        ONNX_OUT    = "results/kaggle/jenkins_build_${BUILD_NUMBER}.onnx"
        // Số episodes ngắn để demo — đủ verify pipeline, không cần convergence
        DEMO_EPISODES = '100'
    }

    stages {
        // ── 1. Setup môi trường ──────────────────────────
        stage('Setup') {
            steps {
                echo "Agent: ${NODE_NAME}"
                sh '''
                    python3 --version
                    pip install --quiet -r requirements-train.txt
                    mkdir -p results/kaggle results/models
                '''
            }
        }

        // ── 2. Sanity check env ──────────────────────────
        stage('Env Check') {
            steps {
                sh '''
                    python3 -c "
import sys
sys.path.insert(0, '.')
from envs.spot_orchestrator_env import SpotOrchestratorEnv
from envs.action_schema import N_ACTIONS
env = SpotOrchestratorEnv()
obs, _ = env.reset()
print(f'OK: obs_dim={len(obs)} n_actions={N_ACTIONS}')
"
                '''
            }
        }

        // ── 3. Short training ────────────────────────────
        stage('Train') {
            steps {
                echo "Training ${DEMO_EPISODES} episodes on ${NODE_NAME}..."
                sh '''
                    python3 -c "
import sys, os, torch
sys.path.insert(0, '.')
from envs.spot_orchestrator_env import SpotOrchestratorEnv
from agents.dqn_agent import DQNAgent

env = SpotOrchestratorEnv()
obs, _ = env.reset()
agent = DQNAgent(state_dim=len(obs), action_dim=env.action_space.n)

best_reward = float('-inf')
for ep in range(int(os.environ['DEMO_EPISODES'])):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.train_step()
        obs = next_obs
        total_reward += reward
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'episode': ep,
            'reward': best_reward,
        }, os.environ['MODEL_OUT'])
    if (ep+1) % 10 == 0:
        print(f'ep {ep+1}/{os.environ[\"DEMO_EPISODES\"]}  reward={total_reward:.2f}  best={best_reward:.2f}')

print(f'Done. Best reward: {best_reward:.2f}')
print(f'Model saved: {os.environ[\"MODEL_OUT\"]}')
"
                '''
            }
        }

        // ── 4. Export ONNX ───────────────────────────────
        stage('Export ONNX') {
            steps {
                sh '''
                    python3 scripts/export_model.py \
                        --model  "$MODEL_OUT" \
                        --output "$ONNX_OUT"
                    echo "ONNX exported: $ONNX_OUT"
                    ls -lh "$ONNX_OUT"
                '''
            }
        }

        // ── 5. Validate ONNX ─────────────────────────────
        stage('Validate ONNX') {
            steps {
                sh '''
                    python3 -c "
import onnxruntime as ort
import numpy as np, os, sys
sys.path.insert(0, '.')
from envs.instance_catalog import STATE_DIM

sess = ort.InferenceSession(os.environ['ONNX_OUT'])
dummy = np.zeros((1, STATE_DIM), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(f'ONNX OK: output shape={out[0].shape}')
"
                '''
            }
        }
    }

    post {
        success {
            // Archive artifacts để download từ Jenkins UI
            archiveArtifacts artifacts: "results/kaggle/jenkins_build_${BUILD_NUMBER}.*",
                             fingerprint: true

            echo """
=== Build ${BUILD_NUMBER} SUCCESS ===
Model : ${MODEL_OUT}
ONNX  : ${ONNX_OUT}
Agent : ${NODE_NAME}

Deploy ONNX lên controller:
"""
        }
        failure {
            echo "Build ${BUILD_NUMBER} FAILED on agent ${NODE_NAME}"
        }
        always {
            cleanWs(patterns: [[pattern: 'venv/**', type: 'INCLUDE']])
        }
    }
}
