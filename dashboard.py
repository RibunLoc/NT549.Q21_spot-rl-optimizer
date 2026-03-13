"""
Streamlit dashboard for Spot RL results.
"""

from pathlib import Path
import json
import pandas as pd
import streamlit as st


RESULTS_DIR = Path("results")
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR = RESULTS_DIR / "plots"


def list_files(path: Path):
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_file()])


def render_report_file(path: Path):
    if path.suffix.lower() in [".md", ".txt"]:
        st.markdown(path.read_text(encoding="utf-8"))
        return
    if path.suffix.lower() == ".json":
        st.json(json.loads(path.read_text(encoding="utf-8")))
        return
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        st.dataframe(df)
        return
    st.write(f"Unsupported file type: {path.name}")


st.set_page_config(page_title="Spot RL Dashboard", layout="wide")
st.title("Spot RL Dashboard")

st.sidebar.header("Artifacts")
report_files = list_files(REPORTS_DIR)
plot_files = list_files(PLOTS_DIR)

if not report_files and not plot_files:
    st.warning("No reports or plots found in results/.")

if report_files:
    selected_report = st.sidebar.selectbox("Report file", report_files)
else:
    selected_report = None

if selected_report:
    st.subheader(f"Report: {selected_report}")
    render_report_file(REPORTS_DIR / selected_report)

st.subheader("Baseline Comparison")
baseline_csv = REPORTS_DIR / "baseline_comparison.csv"
if baseline_csv.exists():
    st.dataframe(pd.read_csv(baseline_csv))
else:
    st.info("baseline_comparison.csv not found.")

if plot_files:
    st.subheader("Plots")
    for plot_name in plot_files:
        if plot_name.lower().endswith((".png", ".jpg", ".jpeg")):
            st.image(str(PLOTS_DIR / plot_name), caption=plot_name, use_column_width=True)
