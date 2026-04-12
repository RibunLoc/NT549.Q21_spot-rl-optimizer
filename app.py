"""
FastAPI service for Spot RL — multi-pool model serving.

Endpoints:
  - GET  /health
  - POST /predict   — get action recommendation from DQN model
  - GET  /models    — list available models
  - GET  /runs      — list training runs
"""

import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent))
from agents.dqn_agent import DQNAgent
from envs.instance_catalog import (
    STATE_DIM, N_ACTIONS, N_TYPES, N_AZS,
    OP_NAMES, INSTANCE_TYPES, AVAILABILITY_ZONES,
    decode_action,
)

app = FastAPI(title="Spot RL API — Multi-Pool", version="1.0.0")

RESULTS_DIR = Path("results")

# Best model from latest training run
DEFAULT_MODEL = RESULTS_DIR / "mp_stable_v2" / "20260410_164435" / "models" / "best_model.pth"

# ── Global model cache ──────────────────────────────────────
_agent: DQNAgent | None = None


def _load_agent(model_path: Path) -> DQNAgent:
    agent = DQNAgent(
        state_dim=STATE_DIM,    # 33
        action_dim=N_ACTIONS,   # 105
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1,
    )
    agent.load(str(model_path))
    agent.epsilon = 0.0
    return agent


def _get_agent() -> DQNAgent:
    global _agent
    if _agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /models/load first.")
    return _agent


# ── Request / Response schemas ──────────────────────────────
class PredictRequest(BaseModel):
    state: List[float]          # 33-feature normalized vector

class PredictResponse(BaseModel):
    action: int
    op: str
    instance_type: str
    az: str
    q_values: List[float]
    confidence: float


class LoadModelRequest(BaseModel):
    path: str


# ── Startup ─────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global _agent
    if DEFAULT_MODEL.exists():
        _agent = _load_agent(DEFAULT_MODEL)
        print(f"Loaded model: {DEFAULT_MODEL}")
    else:
        print(f"Warning: default model not found: {DEFAULT_MODEL}")


# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health() -> Dict:
    return {
        "status": "ok",
        "model_loaded": _agent is not None,
        "state_dim": STATE_DIM,
        "action_dim": N_ACTIONS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Get action recommendation from DQN model.

    Input:  state — 33-feature normalized vector
    Output: recommended (op, instance_type, az) + Q-values + confidence
    """
    agent = _get_agent()

    if len(req.state) != STATE_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {STATE_DIM} features, got {len(req.state)}"
        )

    state = np.array(req.state, dtype=np.float32)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_vals = agent.q_network(state_tensor).squeeze(0).cpu().numpy()

    action = int(q_vals.argmax())
    op, type_idx, az_idx = decode_action(action)

    exp_q = np.exp(q_vals - q_vals.max())
    confidence = float((exp_q / exp_q.sum())[action])

    return PredictResponse(
        action=action,
        op=OP_NAMES[op],
        instance_type=INSTANCE_TYPES[type_idx].name,
        az=AVAILABILITY_ZONES[az_idx],
        q_values=q_vals.tolist(),
        confidence=confidence,
    )


@app.get("/models")
def list_models() -> Dict:
    """List available model checkpoints."""
    models = []
    for run_dir in sorted(RESULTS_DIR.glob("*/*/models")):
        for f in sorted(run_dir.glob("*.pth")):
            models.append(str(f.relative_to(RESULTS_DIR)))
    return {
        "default": str(DEFAULT_MODEL.relative_to(RESULTS_DIR)) if DEFAULT_MODEL.exists() else None,
        "loaded": _agent is not None,
        "available": models,
    }


@app.post("/models/load")
def load_model(req: LoadModelRequest):
    """Load a specific model checkpoint."""
    global _agent
    model_path = Path(req.path)
    if not model_path.is_absolute():
        model_path = RESULTS_DIR / model_path
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    _agent = _load_agent(model_path)
    return {"status": "ok", "loaded": str(model_path)}


@app.get("/runs")
def list_runs() -> Dict:
    """List all training runs with metrics summary."""
    runs = []
    for metrics_file in sorted(RESULTS_DIR.glob("*/*/metrics.pkl")):
        try:
            import pickle, numpy as np
            with open(metrics_file, "rb") as f:
                m = pickle.load(f)
            rewards = m.get("episode_rewards", [])
            runs.append({
                "run": metrics_file.parent.parent.name,
                "timestamp": metrics_file.parent.name,
                "episodes": len(rewards),
                "best_reward": float(max(rewards)) if rewards else None,
                "final_avg_reward": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else None,
            })
        except Exception:
            runs.append({"run": str(metrics_file), "error": "failed to load"})
    return {"runs": runs}