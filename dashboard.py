"""
Streamlit dashboard for Spot RL Multi-Pool — monitoring & live prediction.

Usage:
    streamlit run dashboard.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.append(str(Path(__file__).parent))
from agents.dqn_agent import DQNAgent
from envs.instance_catalog import (
    STATE_DIM, N_ACTIONS, N_TYPES, N_AZS,
    OP_NAMES, INSTANCE_TYPES, AVAILABILITY_ZONES,
    decode_action,
)

RESULTS_DIR = Path("results")
DEFAULT_MODEL = RESULTS_DIR / "mp_stable_v2" / "20260410_164435" / "models" / "best_model.pth"
DEFAULT_METRICS = RESULTS_DIR / "mp_stable_v2" / "20260410_164435" / "metrics.pkl"

# ── Page config ─────────────────────────────────────────────
st.set_page_config(page_title="Spot RL Dashboard", layout="wide", page_icon="⚡")
st.title("⚡ Spot RL Multi-Pool Optimization Dashboard")

page = st.sidebar.radio("Navigation", [
    "Training Curves",
    "Live Prediction",
    "Model Info",
])


# ── Helpers ─────────────────────────────────────────────────
@st.cache_resource
def load_agent(model_path: str) -> DQNAgent:
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1,
    )
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent


@st.cache_data
def load_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "rb") as f:
        return pickle.load(f)


def find_metrics_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*/*/metrics.pkl"))


def find_model_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*/*/models/best_model.pth"))


# ═══════════════════════════════════════════════════════════
# PAGE: Training Curves
# ═══════════════════════════════════════════════════════════
if page == "Training Curves":
    st.header("Training Curves")

    metrics_files = find_metrics_files()
    if not metrics_files:
        st.warning("No metrics.pkl found under results/. Run training first.")
        st.stop()

    options = {f"{f.parent.parent.name}/{f.parent.name}": f for f in metrics_files}
    selected = st.selectbox("Training Run", list(options.keys()),
                            index=len(options) - 1)
    m = load_metrics(str(options[selected]))

    rewards = m.get("episode_rewards", [])
    costs   = m.get("episode_costs", [])
    slas    = m.get("episode_sla_compliance", [])
    n_ep    = len(rewards)

    if n_ep == 0:
        st.error("No episode data found in metrics.")
        st.stop()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Episodes", n_ep)
    col2.metric("Peak Reward", f"{max(rewards):+.1f}")
    col3.metric("Final Avg Reward (100-ep)", f"{np.mean(rewards[-100:]):+.1f}")
    col4.metric("Final Avg SLA", f"{np.mean(slas[-100:]):.1%}" if slas else "N/A")

    st.divider()

    # Smoothed curves
    window = st.slider("Smoothing window", 10, 200, 50)
    df = pd.DataFrame({
        "Episode": range(1, n_ep + 1),
        "Reward":  rewards,
        "Cost":    costs,
        "SLA":     slas,
    }).set_index("Episode")

    df_smooth = df.rolling(window, min_periods=1).mean()

    metric = st.selectbox("Metric", ["Reward", "Cost", "SLA"])
    st.line_chart(df_smooth[metric])
    st.caption(f"{n_ep} episodes — smoothing window={window}")


# ═══════════════════════════════════════════════════════════
# PAGE: Live Prediction
# ═══════════════════════════════════════════════════════════
elif page == "Live Prediction":
    st.header("Live Model Prediction")

    model_files = find_model_files()
    if not model_files:
        st.warning("No best_model.pth found. Run training first.")
        st.stop()

    options = {f"{f.parent.parent.parent.name}/{f.parent.parent.name}": f
               for f in model_files}
    # Default to latest
    default_idx = list(options.keys()).index(
        f"mp_stable_v2/20260410_164435"
    ) if "mp_stable_v2/20260410_164435" in options else len(options) - 1

    selected = st.selectbox("Model", list(options.keys()), index=default_idx)
    agent = load_agent(str(options[selected]))

    st.divider()
    st.subheader(f"State Input — {STATE_DIM} features")

    # ── Price features (per cheapest 3 pools) ───────────────
    with st.expander("Price Features (top-3 cheapest pools)", expanded=True):
        cols = st.columns(3)
        prices, interrupt_probs = [], []
        for i in range(3):
            with cols[i]:
                st.markdown(f"**Pool {i+1}**")
                prices.append(st.slider(f"Spot price {i+1} (norm)", 0.0, 1.0, 0.3 + i*0.1, key=f"p{i}"))
                interrupt_probs.append(st.slider(f"Interrupt prob {i+1}", 0.0, 1.0, 0.05, key=f"ip{i}"))

    # ── Aggregate features ───────────────────────────────────
    with st.expander("Aggregate & Infrastructure", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_spot_price  = st.slider("Avg spot price (norm)", 0.0, 1.0, 0.35)
            price_std       = st.slider("Price std (norm)", 0.0, 1.0, 0.1)
            cheapest_az_id  = st.slider("Cheapest AZ id (norm)", 0.0, 1.0, 0.0)
            az_concentration= st.slider("AZ concentration", 0.0, 1.0, 0.4)
        with c2:
            avg_od_price    = st.slider("Avg OD price (norm)", 0.0, 1.0, 0.6)
            type_price_std  = st.slider("Type price std (norm)", 0.0, 1.0, 0.2)
            cheapest_type_id= st.slider("Cheapest type id (norm)", 0.0, 1.0, 0.0)
            total_spot      = st.slider("Total spot instances (norm)", 0.0, 1.0, 0.5)
        with c3:
            total_od        = st.slider("Total OD instances (norm)", 0.0, 1.0, 0.1)
            total_instances = st.slider("Total instances (norm)", 0.0, 1.0, 0.5)
            total_vcpu      = st.slider("Total vCPU (norm)", 0.0, 1.0, 0.5)
            spot_ratio      = st.slider("Spot ratio", 0.0, 1.0, 0.8)

    # ── Workload & Time ──────────────────────────────────────
    with st.expander("Workload & Time", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            pending_norm    = st.slider("Pending jobs (norm)", 0.0, 1.0, 0.3)
            running_norm    = st.slider("Running jobs (norm)", 0.0, 1.0, 0.4)
            util_ratio      = st.slider("Utilization ratio", 0.0, 1.0, 0.4)
        with c2:
            hour_sin        = st.slider("Hour sin", -1.0, 1.0, 0.0)
            hour_cos        = st.slider("Hour cos", -1.0, 1.0, 1.0)
            day_sin         = st.slider("Day sin", -1.0, 1.0, 0.0)
        with c3:
            day_cos         = st.slider("Day cos", -1.0, 1.0, 1.0)
            ep_progress     = st.slider("Episode progress", 0.0, 1.0, 0.1)
            step_cost_norm  = st.slider("Step cost (norm)", 0.0, 1.0, 0.3)

    state = np.array([
        *prices, *interrupt_probs,                              # [0-5]  top-3 pools
        avg_spot_price, price_std, cheapest_az_id,             # [6-8]
        az_concentration, avg_od_price, type_price_std,        # [9-11]
        cheapest_type_id, total_spot, total_od,                # [12-14]
        total_instances, total_vcpu, spot_ratio,               # [15-17]
        pending_norm, running_norm, util_ratio,                # [18-20]
        hour_sin, hour_cos, day_sin, day_cos,                  # [21-24]
        ep_progress, step_cost_norm,                           # [25-26]
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,                        # [27-32] padding to 33
    ], dtype=np.float32)[:STATE_DIM]

    # ── Predict ─────────────────────────────────────────────
    with torch.no_grad():
        t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_vals = agent.q_network(t).squeeze(0).cpu().numpy()

    # Top action per operation
    action = int(q_vals.argmax())
    op, type_idx, az_idx = decode_action(action)
    exp_q = np.exp(q_vals - q_vals.max())
    probs = exp_q / exp_q.sum()

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Operation",      OP_NAMES[op])
    c2.metric("Instance Type",  INSTANCE_TYPES[type_idx].name)
    c3.metric("AZ",             AVAILABILITY_ZONES[az_idx])
    c4.metric("Confidence",     f"{probs[action]:.1%}")

    # Q-values by operation (max over type×az)
    st.subheader("Q-values by Operation")
    op_q = np.array([q_vals[o * N_TYPES * N_AZS:(o+1) * N_TYPES * N_AZS].max()
                     for o in range(len(OP_NAMES))])
    qdf = pd.DataFrame({"Operation": OP_NAMES, "Max Q-Value": op_q}).set_index("Operation")
    st.bar_chart(qdf)


# ═══════════════════════════════════════════════════════════
# PAGE: Model Info
# ═══════════════════════════════════════════════════════════
elif page == "Model Info":
    st.header("Model & Environment Info")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Environment")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| State dim | {STATE_DIM} |
| Action dim | {N_ACTIONS} |
| Ops | {len(OP_NAMES)} |
| Instance types | {N_TYPES} |
| AZs | {N_AZS} |
        """)

        st.subheader("Operations")
        for i, op in enumerate(OP_NAMES):
            st.write(f"`{i}` {op}")

    with col2:
        st.subheader("Instance Types")
        rows = [{"Type": t.name, "vCPU": t.vcpus,
                 "Memory (GB)": t.memory_gb,
                 "OD Price ($/hr)": t.ondemand_price}
                for t in INSTANCE_TYPES]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Availability Zones")
        for az in AVAILABILITY_ZONES:
            st.write(f"• {az}")

    st.divider()
    st.subheader("Available Training Runs")
    metrics_files = find_metrics_files()
    if metrics_files:
        run_data = []
        for f in metrics_files:
            try:
                m = load_metrics(str(f))
                rewards = m.get("episode_rewards", [])
                run_data.append({
                    "Run": f.parent.parent.name,
                    "Timestamp": f.parent.name,
                    "Episodes": len(rewards),
                    "Peak Reward": f"{max(rewards):+.1f}" if rewards else "N/A",
                    "Final Avg (100-ep)": f"{np.mean(rewards[-100:]):+.1f}" if len(rewards) >= 100 else "N/A",
                })
            except Exception:
                pass
        st.dataframe(pd.DataFrame(run_data), use_container_width=True)
    else:
        st.warning("No training runs found.")
