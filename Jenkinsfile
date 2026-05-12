// CI/CD pipeline — chạy trên Spot agent (label: spot-worker)
// Stages: setup → data pull → env check → resume train → export ONNX → validate → archive
pipeline {
    agent { label 'spot-worker' }

    options {
        timeout(time: 30, unit: 'MINUTES')
        retry(1)
        buildDiscarder(logRotator(numToKeepStr: '5'))
    }

    environment {
        RESULTS_DIR    = 'results/kaggle'
        CHECKPOINT     = 'results/kaggle/best_mixed_v27.pth'   // base model từ DVC
        MODEL_OUT      = "results/kaggle/jenkins_build_${BUILD_NUMBER}.pth"
        ONNX_OUT       = "results/kaggle/jenkins_build_${BUILD_NUMBER}.onnx"
        DEMO_EPISODES  = '50'   // resume thêm 50 ep — đủ demo, không lâu
    }

    stages {
        // ── 1. Setup ─────────────────────────────────────
        stage('Setup') {
            steps {
                echo "Agent: ${NODE_NAME}"
                sh '''
                    python3 --version
                    mkdir -p results/kaggle results/models
                    python3 -m venv .venv
                    .venv/bin/pip install --quiet -r requirements-train.txt
                    .venv/bin/pip install --quiet dvc dvc-s3
                '''
            }
        }

        // ── 2. Pull data + model từ DVC (S3) ─────────────
        stage('DVC Pull') {
            steps {
                sh '''
                    # Pull cả data lẫn model checkpoint v27
                    .venv/bin/dvc pull data/raw/spot_prices/ \
                                       data/processed/ \
                                       results/kaggle/best_mixed_v27.pth
                    echo "Checkpoint: $(ls -lh $CHECKPOINT)"
                    echo "Data files: $(ls data/processed/*.parquet | wc -l) parquet files"
                '''
            }
        }

        // ── 3. Env check ─────────────────────────────────
        stage('Env Check') {
            steps {
                sh '''
                    .venv/bin/python3 -c "
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

        // ── 4. Resume training từ v27 checkpoint ─────────
        stage('Train') {
            steps {
                echo "Resuming from ${CHECKPOINT}, +${DEMO_EPISODES} episodes on ${NODE_NAME}..."
                sh '''
                    .venv/bin/python3 -c "
import sys, os, torch, shutil
sys.path.insert(0, '.')
from envs.spot_orchestrator_env import SpotOrchestratorEnv
from agents.dqn_agent import DQNAgent

env = SpotOrchestratorEnv()
obs, _ = env.reset()
agent = DQNAgent(state_dim=len(obs), action_dim=env.action_space.n)

# Load v27 checkpoint
ckpt = torch.load(os.environ['CHECKPOINT'], map_location='cpu', weights_only=False)
agent.q_network.load_state_dict(ckpt['q_network_state_dict'])
agent.target_network.load_state_dict(ckpt['q_network_state_dict'])
# Giữ epsilon thấp — đã trained, chỉ fine-tune
agent.epsilon = 0.05
print(f'Loaded checkpoint, resuming with epsilon={agent.epsilon}')

best_reward = float('-inf')
for ep in range(int(os.environ['DEMO_EPISODES'])):
    obs, _ = env.reset()
    total_reward, done = 0, False
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
        torch.save({'q_network_state_dict': agent.q_network.state_dict(),
                    'episode': ep, 'reward': best_reward},
                   os.environ['MODEL_OUT'])
    if (ep+1) % 10 == 0:
        print(f'ep {ep+1}/{os.environ[\"DEMO_EPISODES\"]}  reward={total_reward:.2f}  best={best_reward:.2f}')

print(f'Done. best_reward={best_reward:.2f}  saved={os.environ[\"MODEL_OUT\"]}')
"
                '''
            }
        }

        // ── 5. Evaluate model ────────────────────────────
        stage('Evaluate') {
            steps {
                sh '''
                    EVAL_DIR="results/kaggle/eval_build_${BUILD_NUMBER}"
                    mkdir -p "$EVAL_DIR"
                    .venv/bin/python3 experiments/evaluate_kaggle.py \
                        --model "$MODEL_OUT" \
                        --episodes 10 \
                        --output-dir "$EVAL_DIR"
                    echo "=== Eval results ==="
                    ls "$EVAL_DIR/"
                    # In summary nếu có file json
                    find "$EVAL_DIR" -name "*.json" | head -1 | xargs cat 2>/dev/null || true
                '''
            }
        }

        // ── 7. Export ONNX ────────────────────────────────
        stage('Export ONNX') {
            steps {
                sh '''
                    .venv/bin/python3 scripts/export_model.py \
                        --model  "$MODEL_OUT" \
                        --output "$ONNX_OUT"
                    ls -lh "$ONNX_OUT"
                '''
            }
        }

        // ── 8. Validate ONNX ─────────────────────────────
        stage('Validate ONNX') {
            steps {
                sh '''
                    .venv/bin/python3 -c "
import onnxruntime as ort, numpy as np, os, sys
sys.path.insert(0, '.')
from envs.instance_catalog import STATE_DIM

sess = ort.InferenceSession(os.environ['ONNX_OUT'])
dummy = np.zeros((1, STATE_DIM), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(f'ONNX OK: input_dim={STATE_DIM}  output_shape={out[0].shape}')
"
                '''
            }
        }
    }

    post {
        success {
            archiveArtifacts artifacts: "results/kaggle/jenkins_build_${BUILD_NUMBER}.*,results/kaggle/eval_build_${BUILD_NUMBER}/**",
                             fingerprint: true
            echo """
=== Build #${BUILD_NUMBER} SUCCESS ===
Base     : ${CHECKPOINT}  (v27, SLA~96%)
Output   : ${MODEL_OUT}
ONNX     : ${ONNX_OUT}
Agent    : ${NODE_NAME}

Deploy lên controller:
  scp ${ONNX_OUT} ec2-user@<controller-ip>:/opt/spot-rl/model.onnx
"""
        }
        failure {
            echo "Build #${BUILD_NUMBER} FAILED on ${NODE_NAME}"
        }
        always {
            cleanWs(patterns: [[pattern: '.venv/**', type: 'INCLUDE']])
        }
    }
}
