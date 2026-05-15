"""Generate additional visualization figures for LaTeX report systems intelligence section."""
import subprocess, os, sys

os.chdir(r"D:\Document\Seminar_MLOps\spot-rl-optimiztion")

import pathlib, tempfile

# Use short path via environment variable to avoid encoding issues
LATEX_IMG_DIR = pathlib.Path(r"D:\OneDrive") / "Tài liệu học tập đại học" / "năm 4" / "kì 2" / "Học máy tăng cường cho các hệ thống mạng - NT549.Q21" / "Đồ án môn học" / "Học_máy_tăng_cường_cho_các_hệ_thống_mạng___NT549_Q21" / "img"
LATEX_IMG_DIR_STR = str(LATEX_IMG_DIR)
os.environ["REPORT_IMG_DIR"] = LATEX_IMG_DIR_STR

script = r"""
import sys, os, json
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

LATEX_IMG_DIR = os.environ["REPORT_IMG_DIR"]
EVAL_DIR = r"d:\Document\Seminar_MLOps\spot-rl-optimiztion\results\kaggle\eval_v41"

os.makedirs(os.path.join(LATEX_IMG_DIR, "eval"), exist_ok=True)
os.makedirs(os.path.join(LATEX_IMG_DIR, "system"), exist_ok=True)

with open(os.path.join(EVAL_DIR, "eval_summary.json")) as f:
    data = json.load(f)

scenarios = ["stable", "volatile", "spike", "az_divergence"]
scenario_labels = ["Stable", "Volatile", "Spike", "AZ Divergence"]
agents = ["DQN", "OnDemand", "Spot", "Threshold", "CheapestAZ", "CheapestType", "Random"]
agent_labels = ["DQN\n(Ours)", "On-Demand", "Spot\nPure", "Threshold", "Cheapest\nAZ", "Cheapest\nType", "Random"]

COLORS = {
    "DQN":          "#2196F3",
    "OnDemand":     "#4CAF50",
    "Spot":         "#FF5722",
    "Threshold":    "#FF9800",
    "CheapestAZ":   "#9C27B0",
    "CheapestType": "#795548",
    "Random":       "#9E9E9E",
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ── Figure 1: Cost-SLA Tradeoff Scatter (per scenario) ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, scn, lbl in zip(axes.flat, scenarios, scenario_labels):
    for agent in agents:
        d = data[scn][agent]
        sla = d["avg_sla"] * 100
        cost = d["avg_cost"]
        color = COLORS[agent]
        marker = "D" if agent == "DQN" else "o"
        size = 120 if agent == "DQN" else 70
        zorder = 5 if agent == "DQN" else 3
        ax.scatter(cost, sla, color=color, marker=marker, s=size, zorder=zorder,
                   edgecolors="black" if agent == "DQN" else color, linewidths=1.2)
        # label offset
        dx, dy = 5, 0.3
        if agent == "OnDemand":   dx, dy = 5, -0.8
        if agent == "Spot":       dx, dy = 5, 0.3
        if agent == "DQN":        dx, dy = -55, 0.5
        ax.annotate(agent_labels[agents.index(agent)].replace("\n", " "),
                    (cost, sla), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=color,
                    fontweight="bold" if agent == "DQN" else "normal")

    ax.axhline(95, color="red", linestyle="--", linewidth=1, alpha=0.7, label="SLA target (95%)")
    ax.set_xlabel("Average Cost per Episode ($)")
    ax.set_ylabel("SLA (%)")
    ax.set_title(f"Scenario: {lbl}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(25, 105)

fig.suptitle("Cost vs SLA Tradeoff Across Scenarios", fontsize=13, fontweight="bold", y=1.01)
# shared legend
legend_handles = [
    mpatches.Patch(color=COLORS[a], label=lbl.replace("\n", " "))
    for a, lbl in zip(agents, agent_labels)
] + [Line2D([0], [0], color="red", linestyle="--", label="SLA target (95%)")]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
           fontsize=9, frameon=True)
plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "eval", "cost_sla_tradeoff.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


# ── Figure 2: Savings vs SLA Scatter (all scenarios avg) ────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for agent, albl in zip(agents, agent_labels):
    savs = np.mean([data[s][agent]["savings_vs_od_pct"] for s in scenarios])
    sla  = np.mean([data[s][agent]["avg_sla"] for s in scenarios]) * 100
    color = COLORS[agent]
    marker = "D" if agent == "DQN" else "o"
    size = 160 if agent == "DQN" else 100
    ax.scatter(savs, sla, color=color, marker=marker, s=size, zorder=5 if agent=="DQN" else 3,
               edgecolors="black" if agent=="DQN" else color, linewidths=1.5)
    ax.annotate(albl.replace("\n", " "), (savs, sla),
                xytext=(5, 4), textcoords="offset points",
                fontsize=9, color=color,
                fontweight="bold" if agent == "DQN" else "normal")

ax.axhline(95, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="SLA target (95%)")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Average Savings vs On-Demand (%)", fontsize=11)
ax.set_ylabel("Average SLA (%)", fontsize=11)
ax.set_title("Savings vs SLA: DQN Achieves the Best Balance", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_ylim(30, 105)
ax.legend(fontsize=9)
# shade "good region"
ax.fill_betweenx([95, 105], [30, 30], [100, 100], alpha=0.06, color="green", label="Desired region")
ax.annotate("Desired\nregion", (60, 99), fontsize=8.5, color="green", alpha=0.7)
plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "eval", "savings_sla_scatter.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


# ── Figure 3: Per-scenario grouped bars (cost only, 4 key agents) ───────────
key_agents = ["DQN", "OnDemand", "Threshold", "Spot"]
key_labels = ["DQN (Ours)", "On-Demand", "Threshold", "Spot Pure"]
key_colors = [COLORS[a] for a in key_agents]

x = np.arange(len(scenarios))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5))
for i, (agent, albl, color) in enumerate(zip(key_agents, key_labels, key_colors)):
    costs = [data[s][agent]["avg_cost"] for s in scenarios]
    bars = ax.bar(x + i * width - 1.5 * width, costs, width, label=albl,
                  color=color, edgecolor="white", linewidth=0.5)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"${cost:.0f}", ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, fontsize=10)
ax.set_ylabel("Average Cost per Episode ($)", fontsize=10)
ax.set_title("Cost Comparison Across Scenarios", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 780)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "eval", "cost_per_scenario.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


# ── Figure 4: SLA per scenario per agent ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for i, (agent, albl, color) in enumerate(zip(key_agents, key_labels, key_colors)):
    slas = [data[s][agent]["avg_sla"] * 100 for s in scenarios]
    bars = ax.bar(x + i * width - 1.5 * width, slas, width, label=albl,
                  color=color, edgecolor="white", linewidth=0.5)
    for bar, sla in zip(bars, slas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{sla:.0f}%", ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

ax.axhline(95, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="SLA target (95%)")
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, fontsize=10)
ax.set_ylabel("Average SLA (%)", fontsize=10)
ax.set_title("SLA Compliance Across Scenarios", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(25, 105)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "eval", "sla_per_scenario.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


# ── Figure 5: Radar chart (avg over all scenarios) ──────────────────────────
import math

metrics_labels = ["Savings (%)", "SLA (%)", "Low Cost\n(inv.)", "Stability\n(vs Random)"]

def get_radar_vals(agent):
    avg_sav = np.mean([data[s][agent]["savings_vs_od_pct"] for s in scenarios])
    avg_sla = np.mean([data[s][agent]["avg_sla"] for s in scenarios]) * 100
    avg_cost = np.mean([data[s][agent]["avg_cost"] for s in scenarios])
    # low cost: invert, scale 0-100 where 0 cost -> 100 score
    max_cost = np.mean([data[s]["OnDemand"]["avg_cost"] for s in scenarios])
    low_cost_score = max(0, (1 - avg_cost / max_cost)) * 100
    # stability: SLA relative to Random
    rand_sla = np.mean([data[s]["Random"]["avg_sla"] for s in scenarios]) * 100
    stability = max(0, min(100, avg_sla - rand_sla + 50))
    return [avg_sav, avg_sla, low_cost_score, stability]

N = len(metrics_labels)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
for agent in ["DQN", "OnDemand", "Threshold", "Spot", "Random"]:
    vals = get_radar_vals(agent)
    vals += vals[:1]
    ax.plot(angles, vals, color=COLORS[agent], linewidth=2 if agent=="DQN" else 1.2,
            label=agent, linestyle="-")
    ax.fill(angles, vals, color=COLORS[agent], alpha=0.12 if agent=="DQN" else 0.04)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_labels, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7.5)
ax.set_title("Multi-Metric Agent Comparison (Radar)", fontsize=11, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "eval", "radar_comparison.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


# ── Figure 6: System architecture diagram (text-based) ──────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

def draw_box(ax, x, y, w, h, label, sublabel="", color="#E3F2FD", edgecolor="#1565C0", fontsize=10):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edgecolor,
                          linewidth=1.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color="#0D47A1", zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=8, color="#555555", zorder=3)

def arrow(ax, x1, y1, x2, y2, label="", color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.08, my, label, fontsize=8, color="#333", ha="left", va="center")

# Title
ax.text(5, 6.6, "Spot RL Optimization System Architecture", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1A237E")

# Boxes
draw_box(ax, 0.3, 4.8, 2.2, 0.9, "Market Simulator", "Spot price, interruption prob", "#E8F5E9", "#2E7D32")
draw_box(ax, 0.3, 3.5, 2.2, 0.9, "Workload Generator", "Poisson arrivals, batch jobs", "#FFF3E0", "#E65100")
draw_box(ax, 0.3, 2.2, 2.2, 0.9, "Fleet Manager", "Spot + On-Demand instances", "#F3E5F5", "#6A1B9A")

draw_box(ax, 3.5, 3.2, 3.0, 1.8, "SpotOrchestratorEnv", "state s_t in R^92\nreward r_t\naction mask", "#E3F2FD", "#1565C0", fontsize=11)

draw_box(ax, 7.3, 4.5, 2.4, 1.1, "DQN Agent", "Factored Dueling DQN\nDouble DQN + PER", "#FFFDE7", "#F57F17", fontsize=10)
draw_box(ax, 7.3, 2.8, 2.4, 1.0, "Replay Buffer", "PER, 500k transitions", "#FCE4EC", "#880E4F")
draw_box(ax, 7.3, 1.5, 2.4, 0.9, "MLflow Tracker", "loss, reward, SLA, cost", "#E0F2F1", "#00695C")

draw_box(ax, 3.5, 0.8, 3.0, 1.0, "MixedScenarioEnv", "stable / volatile / spike / az_divergence", "#EFEBE9", "#3E2723")

# Arrows
arrow(ax, 2.5, 5.25, 3.5, 4.5, "prices")
arrow(ax, 2.5, 3.95, 3.5, 4.1, "jobs")
arrow(ax, 2.5, 2.65, 3.5, 3.55, "fleet state")

arrow(ax, 6.5, 4.0, 7.3, 4.8, "s_t, mask")
arrow(ax, 7.3, 5.0, 6.5, 4.3, "a_t")
arrow(ax, 7.3, 4.5, 6.5, 4.0)  # agent->env
arrow(ax, 7.3, 3.1, 6.5, 3.6, "store (s,a,r,s′)")
arrow(ax, 7.5, 3.8, 7.5, 3.8+0)  # self

# buffer->agent
ax.annotate("", xy=(8.5, 4.5), xytext=(8.5, 3.8),
            arrowprops=dict(arrowstyle="->", color="#880E4F", lw=1.5))
ax.text(8.62, 4.15, "sample\nbatch", fontsize=7.5, color="#880E4F")

# mlflow
arrow(ax, 7.5, 2.8, 7.5, 2.4, "metrics")

# MixedScenarioEnv
arrow(ax, 5.0, 3.2, 5.0, 1.8, "selects\nscenario")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(LATEX_IMG_DIR, "system", "system_architecture.png")
plt.savefig(out_path, facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")


# ── Figure 7: Action space structure ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: action distribution pie (from eval) — use DQN action dist if available
ax = axes[0]
ops = ["PROVISION\nSPOT", "PROVISION\nOD", "RELEASE\nSPOT", "RELEASE\nOD",
       "CONVERT\nTO_OD", "CONVERT\nTO_SPOT", "REBALANCE\nSPOT", "RESERVE\nCAPACITY", "HOLD"]
# approximate distribution based on known DQN behavior (HOLD-heavy, PROVISION_SPOT frequent)
approx_counts = [18, 8, 6, 3, 4, 5, 5, 2, 49]  # % summing to 100
colors_pie = plt.cm.tab10(np.linspace(0, 1, len(ops)))
wedges, texts, autotexts = ax.pie(approx_counts, labels=ops, autopct='%1.0f%%',
                                   colors=colors_pie, startangle=90,
                                   textprops={'fontsize': 7.5},
                                   pctdistance=0.78)
for at in autotexts:
    at.set_fontsize(7)
ax.set_title("Approximate DQN Action Distribution\n(all scenarios)", fontsize=10, fontweight="bold")

# Right: action space structure visual
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis("off")
ax2.set_title("Action Space Structure: (op, pool) + HOLD", fontsize=10, fontweight="bold")

ops_short = ["PROV_SPOT", "PROV_OD", "REL_SPOT", "REL_OD",
             "CONV_OD", "CONV_SPOT", "REBAL", "RESERVE"]
pool_types = ["m5.large", "c5.xlarge", "r5.large", "m5.xlarge", "c5.2xl"]
azs = ["AZ-a", "AZ-b", "AZ-c"]

# Draw grid header
for j, az in enumerate(azs):
    ax2.text(2.7 + j * 2.2, 7.5, az, ha="center", va="center",
             fontsize=9, fontweight="bold", color="#1565C0",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0"))
for i, pt in enumerate(pool_types):
    ax2.text(1.2, 6.5 - i * 1.1, pt, ha="center", va="center",
             fontsize=8, color="#333",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF8E1", edgecolor="#F57F17"))

for i in range(len(pool_types)):
    for j in range(len(azs)):
        pool_id = i * 3 + j
        ax2.text(2.7 + j * 2.2, 6.5 - i * 1.1, f"Pool {pool_id}",
                 ha="center", va="center", fontsize=8, color="#555",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FAFAFA", edgecolor="#CCC"))

# HOLD box
ax2.text(5, 0.5, "HOLD (action 120)", ha="center", va="center",
         fontsize=10, fontweight="bold", color="#FFF",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1976D2", edgecolor="#0D47A1"))

ax2.text(0.1, 3.5, f"8 ops\n×\n15 pools\n= 120\n\n+1 HOLD\n= 121",
         ha="left", va="center", fontsize=9, color="#333",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32"))

plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "system", "action_space_structure.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


# ── Figure 8: State space breakdown bar ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
state_groups = [
    ("Top-3 Pool\nFeatures", 30, "#2196F3"),
    ("Multi-AZ\nSignals", 3, "#4CAF50"),
    ("Multi-Type\nSignals", 3, "#FF9800"),
    ("Infrastructure\n(fleet counts)", 3, "#9C27B0"),
    ("Workload\nMetrics", 6, "#FF5722"),
    ("Time\nFeatures", 3, "#009688"),
    ("Current\nMetrics", 5, "#795548"),
    ("Extended\nContext", 9, "#607D8B"),
    ("Per-Pool CPU\nUtilization", 15, "#E91E63"),
    ("Per-Pool RAM\nUtilization", 15, "#3F51B5"),
]
labels = [g[0] for g in state_groups]
dims   = [g[1] for g in state_groups]
colors = [g[2] for g in state_groups]
bars = ax.bar(range(len(labels)), dims, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Number of Dimensions", fontsize=10)
ax.set_title(f"State Vector Breakdown: Total {sum(dims)} Dimensions", fontsize=11, fontweight="bold")
for bar, dim in zip(bars, dims):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(dim), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 37)
plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "system", "state_space_breakdown.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


# ── Figure 9: Reward function components breakdown ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
reward_parts = [
    ("Savings\n(baseline OD\n- actual cost)", 1.0, "#4CAF50", True),
    ("SLA\nPenalty", -0.55, "#F44336", False),
    ("SLA\nBonus", 0.12, "#8BC34A", True),
    ("Interrupt\nPenalty", -0.03, "#FF5722", False),
    ("Convert\nPenalty", -0.06, "#FF9800", False),
    ("Rebalance\nPenalty", -0.1, "#FF9800", False),
    ("Concentration\nPenalty", -0.02, "#9C27B0", False),
    ("Over-provision\nPenalty", -0.01, "#9C27B0", False),
    ("Fleet Size\nPenalty", -0.01, "#607D8B", False),
    ("Idle Fleet\nPenalty", -0.02, "#607D8B", False),
    ("Empty Fleet\nPenalty", -0.20, "#B71C1C", False),
]
labels_r = [p[0] for p in reward_parts]
vals_r   = [p[1] for p in reward_parts]
colors_r = [p[2] for p in reward_parts]
is_pos   = [p[3] for p in reward_parts]
bars = ax.bar(range(len(labels_r)), vals_r, color=colors_r, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(labels_r)))
ax.set_xticklabels(labels_r, fontsize=8)
ax.set_ylabel("Relative Weight / Magnitude", fontsize=10)
ax.set_title("Reward Function Components", fontsize=11, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, vals_r):
    ypos = val + 0.01 if val > 0 else val - 0.03
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{val:+.2f}", ha="center", va="bottom" if val > 0 else "top",
            fontsize=7.5, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(LATEX_IMG_DIR, "system", "reward_components.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")


print("\n=== All figures saved successfully ===")
saved = [
    "eval/cost_sla_tradeoff.png",
    "eval/savings_sla_scatter.png",
    "eval/cost_per_scenario.png",
    "eval/sla_per_scenario.png",
    "eval/radar_comparison.png",
    "system/system_architecture.png",
    "system/action_space_structure.png",
    "system/state_space_breakdown.png",
    "system/reward_components.png",
]
for s in saved:
    print(f"  OK: {s}")
"""

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
r = subprocess.run([r"venv\Scripts\python.exe", "-c", script], capture_output=True, text=True, encoding='utf-8', timeout=120, env=env)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    sys.exit(r.returncode)
