"""Fix 3 system figures with accurate data from code."""
import subprocess, os, sys

os.chdir(r"D:\Document\Seminar_MLOps\spot-rl-optimiztion")

import pathlib
LATEX_IMG_DIR = pathlib.Path(r"D:\OneDrive") / "Tài liệu học tập đại học" / "năm 4" / "kì 2" / "Học máy tăng cường cho các hệ thống mạng - NT549.Q21" / "Đồ án môn học" / "Học_máy_tăng_cường_cho_các_hệ_thống_mạng___NT549_Q21" / "img"
os.environ["REPORT_IMG_DIR"] = str(LATEX_IMG_DIR)

script = r"""
import os, sys
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LATEX_IMG_DIR = os.environ["REPORT_IMG_DIR"]
SYS_DIR = os.path.join(LATEX_IMG_DIR, "system")
os.makedirs(SYS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ─── Figure 1: system_architecture.png (fixed) ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("#F8F9FA")

def draw_box(ax, x, y, w, h, label, sublabel="", color="#E3F2FD", edgecolor="#1565C0", fontsize=10):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edgecolor, linewidth=1.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#0D47A1", zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=8, color="#555555", zorder=3)

def arrow(ax, x1, y1, x2, y2, label="", color="#555", lblpos="mid", offset=(0.08, 0)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2 + offset[0], (y1+y2)/2 + offset[1]
        ax.text(mx, my, label, fontsize=8, color="#333", ha="left", va="center")

ax.text(5, 6.6, "Spot RL Optimization System Architecture", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1A237E")

draw_box(ax, 0.3, 4.8, 2.2, 0.9, "Market Simulator", "Spot price, interruption prob", "#E8F5E9", "#2E7D32")
draw_box(ax, 0.3, 3.5, 2.2, 0.9, "Workload Generator", "Poisson arrivals, batch jobs", "#FFF3E0", "#E65100")
draw_box(ax, 0.3, 2.2, 2.2, 0.9, "Fleet Manager", "Spot + On-Demand instances", "#F3E5F5", "#6A1B9A")

draw_box(ax, 3.5, 3.2, 3.0, 1.8, "SpotOrchestratorEnv", "state s_t in R^92\nreward r_t\naction mask",
         "#E3F2FD", "#1565C0", fontsize=11)

draw_box(ax, 7.3, 4.5, 2.4, 1.1, "DQN Agent", "Factored Dueling DQN\nDouble DQN + PER",
         "#FFFDE7", "#F57F17", fontsize=10)
# FIX: 500k -> 300k transitions
draw_box(ax, 7.3, 2.8, 2.4, 1.0, "Replay Buffer", "PER, 300k transitions", "#FCE4EC", "#880E4F")
draw_box(ax, 7.3, 1.3, 2.4, 0.9, "MLflow Tracker", "loss, reward, SLA, cost", "#E0F2F1", "#00695C")

draw_box(ax, 3.5, 0.6, 3.0, 1.0, "MixedScenarioEnv",
         "stable / volatile / spike / az_divergence", "#EFEBE9", "#3E2723")

# Arrows from data sources to env
arrow(ax, 2.5, 5.25, 3.5, 4.5, "prices")
arrow(ax, 2.5, 3.95, 3.5, 4.1, "jobs")
arrow(ax, 2.5, 2.65, 3.5, 3.55, "fleet state")

# Env <-> Agent
arrow(ax, 6.5, 4.3, 7.3, 4.9, "s_t, mask")
arrow(ax, 7.3, 4.7, 6.5, 4.0, "a_t")

# Env -> Replay Buffer
arrow(ax, 6.5, 3.6, 7.3, 3.3, "store (s,a,r,s')")

# Replay Buffer -> Agent (sample)
ax.annotate("", xy=(8.5, 4.5), xytext=(8.5, 3.8),
            arrowprops=dict(arrowstyle="->", color="#880E4F", lw=1.5))
ax.text(8.62, 4.15, "sample\nbatch", fontsize=7.5, color="#880E4F")

# FIX: metrics arrow now from Agent -> MLflow (not from Replay Buffer)
ax.annotate("", xy=(8.5, 2.2), xytext=(8.5, 4.5),
            arrowprops=dict(arrowstyle="->", color="#00695C", lw=1.3,
                            connectionstyle="arc3,rad=0.3"))
ax.text(9.75, 3.4, "metrics", fontsize=8, color="#00695C", ha="left")

# Env -> MLflow (env-side metrics: SLA, cost)
ax.annotate("", xy=(7.3, 1.75), xytext=(6.5, 3.3),
            arrowprops=dict(arrowstyle="->", color="#00695C", lw=1.3, linestyle="--"))
ax.text(6.7, 2.4, "SLA/cost", fontsize=7.5, color="#00695C")

# MixedScenarioEnv -> SpotOrchestratorEnv (wraps it)
ax.annotate("", xy=(5.0, 3.2), xytext=(5.0, 1.6),
            arrowprops=dict(arrowstyle="->", color="#3E2723", lw=1.3))
ax.text(5.1, 2.4, "wraps /\nselects\nscenario", fontsize=7.5, color="#3E2723")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(SYS_DIR, "system_architecture.png")
plt.savefig(out, facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: system_architecture.png")


# ─── Figure 2: action_space_structure.png (no pie chart, just structure) ─────
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 11); ax.set_ylim(0, 8); ax.axis("off")
ax.set_title("Action Space Structure: (op, pool) + HOLD = 121 actions",
             fontsize=13, fontweight="bold", color="#1A237E", pad=15)

# Left: Operations list
ops = [
    ("0", "PROVISION_SPOT",      "Create new Spot instance"),
    ("1", "PROVISION_ONDEMAND",  "Create new OD instance"),
    ("2", "RELEASE_SPOT",        "Terminate Spot instance"),
    ("3", "RELEASE_ONDEMAND",    "Terminate OD instance"),
    ("4", "CONVERT_TO_ONDEMAND", "Spot -> OD (graceful fallback)"),
    ("5", "CONVERT_TO_SPOT",     "OD -> Spot (opportunistic)"),
    ("6", "REBALANCE_SPOT",      "Move Spot to better pool"),
    ("7", "RESERVE_CAPACITY",    "Preemptive OD provisioning"),
]
ax.text(0.5, 7.5, "8 Operations (op)", fontsize=11, fontweight="bold", color="#0D47A1")
for i, (code, name, desc) in enumerate(ops):
    y = 7.0 - i * 0.6
    ax.text(0.5, y, code, fontsize=9, fontweight="bold", color="#FFF",
            bbox=dict(boxstyle="circle,pad=0.25", facecolor="#1565C0", edgecolor="#0D47A1"))
    ax.text(0.95, y, name, fontsize=9, color="#0D47A1", fontweight="bold", va="center")
    ax.text(0.95, y - 0.22, desc, fontsize=7.5, color="#555", va="center")

# Right: 5x3 grid of pools
pool_types = ["m5.large", "c5.xlarge", "r5.large", "m5.xlarge", "c5.2xlarge"]
azs = ["AZ-a\n(ap-se-1a)", "AZ-b\n(ap-se-1b)", "AZ-c\n(ap-se-1c)"]

ax.text(8.5, 7.5, "15 Pools = 5 types x 3 AZs", fontsize=11, fontweight="bold", color="#0D47A1", ha="center")

for j, az in enumerate(azs):
    ax.text(7.3 + j * 1.2, 7.0, az, ha="center", va="center",
            fontsize=8, fontweight="bold", color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#E3F2FD", edgecolor="#1565C0"))
for i, pt in enumerate(pool_types):
    ax.text(6.2, 6.3 - i * 0.7, pt, ha="center", va="center",
            fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF8E1", edgecolor="#F57F17"))

for i in range(5):
    for j in range(3):
        pool_id = i * 3 + j
        ax.text(7.3 + j * 1.2, 6.3 - i * 0.7, f"Pool\n{pool_id}",
                ha="center", va="center", fontsize=7.5, color="#555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FAFAFA", edgecolor="#CCC"))

# Bottom: formula
ax.text(5.5, 1.4,
        r"Action encoding:   $a = \mathrm{op} \times 15 + \mathrm{pool\_id}$    (for non-HOLD)",
        ha="center", fontsize=10, color="#0D47A1",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32"))

# HOLD
ax.text(5.5, 0.5, "HOLD = action 120  (no-op, do not change fleet)",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#FFF",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1976D2", edgecolor="#0D47A1"))

# Total
ax.text(5.5, 2.3, "Total: 8 x 15 + 1 = 121 discrete actions",
        ha="center", fontsize=10.5, fontweight="bold", color="#2E7D32")

plt.tight_layout()
out = os.path.join(SYS_DIR, "action_space_structure.png")
plt.savefig(out)
plt.close()
print(f"Saved: action_space_structure.png")


# ─── Figure 3: reward_components.png — split layout to avoid overlap ────────
# From envs/spot_orchestrator_env.py v41 defaults:
#   sla_penalty=3.5 (per failed job)
#   sla_bonus=0.5   (per step when SLA >= 0.93)
#   interrupt_penalty=0.1
#   convert_to_od_penalty=0.4
#   convert_to_spot_penalty=0.5
#   rebalance_penalty=0.8
#   concentration_penalty=0.05 (threshold 0.8)
#   overprovision_penalty=0.3
#   fleet_size_penalty=0.01
#   idle_fleet_penalty=0.02
#   empty_fleet_penalty=0.5 (per pending job, capped 10)
#   provision_spot_bonus=0.8
import matplotlib.patches as mpatches

# Split into 3 panels: bonuses (left), small penalties (mid), large penalties (right)
fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.08], hspace=0.45, wspace=0.32)
ax_bonus = fig.add_subplot(gs[0, 0])
ax_pen_small = fig.add_subplot(gs[0, 1])
ax_pen_big = fig.add_subplot(gs[0, 2])

fig.suptitle("Reward Function Components (v41 default coefficients)",
             fontsize=13, fontweight="bold", y=0.98)

# Panel 1: Bonuses + Savings reference
bonus_items = [
    ("Savings\n(env-driven)",           1.0, "#4CAF50", True),
    ("Provision Spot\nBonus",           0.8, "#8BC34A", False),
    ("SLA Bonus\n(SLA $\\geq$ 0.93)",   0.5, "#AED581", False),
]
labels_b = [p[0] for p in bonus_items]
vals_b   = [p[1] for p in bonus_items]
colors_b = [p[2] for p in bonus_items]
x_b = np.arange(len(labels_b))
bars_b = ax_bonus.bar(x_b, vals_b, color=colors_b, edgecolor="white", linewidth=0.5, width=0.6)
bars_b[0].set_hatch("//")
bars_b[0].set_edgecolor("#2E7D32")
bars_b[0].set_linewidth(1.5)
for bar, val, item in zip(bars_b, vals_b, bonus_items):
    label = "variable" if item[3] else f"+{val:.2f}"
    ax_bonus.text(bar.get_x() + bar.get_width()/2, val + 0.04, label,
                  ha="center", va="bottom", fontsize=9, fontweight="bold",
                  color="#2E7D32" if item[3] else "#333")
ax_bonus.set_xticks(x_b)
ax_bonus.set_xticklabels(labels_b, fontsize=8.5)
ax_bonus.set_ylabel("Coefficient value", fontsize=9)
ax_bonus.set_title("Positive: Savings + Bonuses", fontsize=10.5, fontweight="bold", color="#2E7D32")
ax_bonus.axhline(0, color="black", linewidth=0.8)
ax_bonus.grid(axis="y", alpha=0.3)
ax_bonus.set_ylim(0, 1.4)

# Panel 2: Small penalties (|coef| <= 0.8)
small_pen = [
    ("Rebalance",       -0.80, "#FF7043"),
    ("Empty Fleet",     -0.50, "#E64A19"),
    ("Convert to Spot", -0.50, "#FF9800"),
    ("Convert to OD",   -0.40, "#FFA726"),
    ("Over-provision",  -0.30, "#FFB74D"),
    ("Interrupt",       -0.10, "#FFCC80"),
    ("Concentration",   -0.05, "#9C27B0"),
    ("Idle Fleet",      -0.02, "#607D8B"),
    ("Fleet Size",      -0.01, "#90A4AE"),
]
labels_s = [p[0] for p in small_pen]
vals_s   = [p[1] for p in small_pen]
colors_s = [p[2] for p in small_pen]
x_s = np.arange(len(labels_s))
bars_s = ax_pen_small.barh(x_s, vals_s, color=colors_s, edgecolor="white", linewidth=0.5, height=0.7)
for bar, val in zip(bars_s, vals_s):
    label = f"{val:+.2f}" if abs(val) >= 0.1 else f"{val:+.3f}"
    ax_pen_small.text(val - 0.02, bar.get_y() + bar.get_height()/2, label,
                       ha="right", va="center", fontsize=8.5, fontweight="bold")
ax_pen_small.set_yticks(x_s)
ax_pen_small.set_yticklabels(labels_s, fontsize=8.5)
ax_pen_small.invert_yaxis()
ax_pen_small.set_xlabel("Coefficient", fontsize=9)
ax_pen_small.set_title("Small penalties ($|c| \\leq 0.8$)", fontsize=10.5, fontweight="bold", color="#E65100")
ax_pen_small.axvline(0, color="black", linewidth=0.8)
ax_pen_small.grid(axis="x", alpha=0.3)
ax_pen_small.set_xlim(-1.0, 0.05)

# Panel 3: Dominant SLA Penalty
big_pen = [("SLA Penalty\n(per failed job)", -3.50, "#D32F2F")]
ax_pen_big.bar([0], [-3.5], color="#D32F2F", edgecolor="#B71C1C", linewidth=1.5, width=0.5)
ax_pen_big.text(0, -3.5 - 0.15, "-3.50", ha="center", va="top", fontsize=12, fontweight="bold", color="#B71C1C")
ax_pen_big.text(0, -1.5, "Dominant penalty\nmultiplied by\n# of failed jobs\nper step",
                ha="center", va="center", fontsize=9, color="#B71C1C",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#D32F2F"))
ax_pen_big.set_xticks([0])
ax_pen_big.set_xticklabels(["SLA Penalty"], fontsize=9.5, fontweight="bold")
ax_pen_big.set_title("Dominant penalty (SLA)", fontsize=10.5, fontweight="bold", color="#B71C1C")
ax_pen_big.axhline(0, color="black", linewidth=0.8)
ax_pen_big.set_ylim(-4.2, 0.3)
ax_pen_big.grid(axis="y", alpha=0.3)
ax_pen_big.set_xlim(-0.6, 0.6)

# Footer with formula (separate row, no overlap)
ax_footer = fig.add_subplot(gs[1, :])
ax_footer.axis("off")
ax_footer.text(0.5, 0.5,
    "Formula:   $R_t = \\mathrm{savings}_t + \\sum_i \\mathrm{bonus}_i - \\sum_j \\mathrm{penalty}_j$"
    "   where   $\\mathrm{savings}_t = \\mathrm{baseline\\_OD\\_cost}_t - \\mathrm{actual\\_cost}_t$",
    ha="center", va="center", fontsize=10.5, color="#1A237E",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8EAF6", edgecolor="#3949AB"))

out = os.path.join(SYS_DIR, "reward_components.png")
plt.savefig(out)
plt.close()
print(f"Saved: reward_components.png")

print("\nAll 3 system figures regenerated with accurate data.")
"""

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
r = subprocess.run([r"venv\Scripts\python.exe", "-c", script], capture_output=True, text=True,
                   encoding="utf-8", timeout=120, env=env)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    sys.exit(r.returncode)
