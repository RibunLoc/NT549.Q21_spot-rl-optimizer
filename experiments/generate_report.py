"""
Generate a static Markdown report from evaluation results.

Usage:
    python experiments/generate_report.py --results-dir results
"""

import argparse
from pathlib import Path
import json
import pandas as pd


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_"

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.itertuples(index=False):
        values = [str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate static report')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory (default: results)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path (default: results/reports/REPORT.md)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    reports_dir = results_dir / 'reports'
    plots_dir = results_dir / 'plots'
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else reports_dir / 'REPORT.md'

    lines = []
    lines.append("# Spot RL Evaluation Report")
    lines.append("")

    # Evaluation summaries
    eval_files = sorted(reports_dir.glob('eval_*.json'))
    if eval_files:
        lines.append("## DQN Evaluation Summary")
        lines.append("")
        for f in eval_files:
            scenario = f.stem.replace('eval_', '')
            with open(f, 'r') as fh:
                summary = json.load(fh)
            df = pd.DataFrame([{
                'scenario': scenario,
                'avg_cost': summary.get('avg_cost', 0.0),
                'cost_savings_pct': summary.get('cost_savings_pct', 0.0),
                'avg_sla': summary.get('avg_sla_compliance', 0.0),
                'sla_violation_rate_pct': summary.get('sla_violation_rate_pct', 0.0),
                'avg_reward': summary.get('avg_reward', 0.0),
            }])
            lines.append(format_table(df))
            lines.append("")

            # Plots
            cost_plot = plots_dir / f"cost_sla_{scenario}.png"
            action_plot = plots_dir / f"action_distribution_{scenario}.png"
            if cost_plot.exists():
                lines.append(f"![Cost/SLA - {scenario}](../plots/{cost_plot.name})")
                lines.append("")
            if action_plot.exists():
                lines.append(f"![Action Distribution - {scenario}](../plots/{action_plot.name})")
                lines.append("")
    else:
        lines.append("## DQN Evaluation Summary")
        lines.append("")
        lines.append("No evaluation summaries found.")
        lines.append("")

    # Baseline comparison
    comparison_csv = reports_dir / 'baseline_comparison.csv'
    if comparison_csv.exists():
        lines.append("## Baseline Comparison")
        lines.append("")
        comp_df = pd.read_csv(comparison_csv)
        for scenario in sorted(comp_df['scenario'].unique()):
            lines.append(f"### Scenario: {scenario}")
            lines.append("")
            subset = comp_df[comp_df['scenario'] == scenario][[
                'agent_name', 'avg_cost', 'cost_savings_pct', 'avg_sla_compliance'
            ]].rename(columns={'avg_sla_compliance': 'avg_sla'})
            lines.append(format_table(subset))
            lines.append("")

            compare_plot = plots_dir / f"cost_sla_comparison_{scenario}.png"
            if compare_plot.exists():
                lines.append(f"![Baseline Comparison - {scenario}](../plots/{compare_plot.name})")
                lines.append("")
    else:
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append("No baseline comparison found.")
        lines.append("")

    # Write report
    output_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"Report generated: {output_path}")


if __name__ == '__main__':
    main()
