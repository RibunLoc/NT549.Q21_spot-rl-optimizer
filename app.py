"""
Minimal FastAPI service for Spot RL artifacts.

Endpoints:
  - GET /health
  - GET /reports
  - GET /plots
  - GET /report/{name}
  - GET /plot/{name}
  - GET /runs
"""

from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

app = FastAPI(title="Spot RL API", version="0.1.0")

RESULTS_DIR = Path("results")
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR = RESULTS_DIR / "plots"


def _safe_name(name: str) -> bool:
    return "/" not in name and "\\" not in name and ".." not in name


def _list_files(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_file()])


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/reports")
def list_reports() -> Dict[str, List[str]]:
    return {"reports": _list_files(REPORTS_DIR)}


@app.get("/plots")
def list_plots() -> Dict[str, List[str]]:
    return {"plots": _list_files(PLOTS_DIR)}


@app.get("/report/{name}")
def get_report(name: str):
    if not _safe_name(name):
        raise HTTPException(status_code=400, detail="Invalid report name")
    report_path = REPORTS_DIR / name
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    if report_path.suffix.lower() in [".md", ".txt"]:
        return PlainTextResponse(report_path.read_text(encoding="utf-8"))
    return FileResponse(report_path)


@app.get("/plot/{name}")
def get_plot(name: str):
    if not _safe_name(name):
        raise HTTPException(status_code=400, detail="Invalid plot name")
    plot_path = PLOTS_DIR / name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(plot_path)


@app.get("/runs")
def list_runs() -> Dict[str, List[str]]:
    if not RESULTS_DIR.exists():
        return {"runs": []}
    runs = []
    for p in RESULTS_DIR.iterdir():
        if not p.is_dir():
            continue
        if p.name in ["models", "reports", "plots", "logs"]:
            continue
        runs.append(p.name)
    return {"runs": sorted(runs)}
