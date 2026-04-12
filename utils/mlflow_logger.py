"""
MLflow integration for experiment tracking.

Usage:
    from utils.mlflow_logger import MLflowLogger

    mlf = MLflowLogger(experiment_name="spot-rl")
    mlf.start_run(run_name="dqn_stable_v5")
    mlf.log_params(config)
    mlf.log_metrics({"reward": 100, "sla": 0.98}, step=10)
    mlf.log_model("path/to/model.pth")
    mlf.end_run()
"""

import mlflow
import mlflow.pytorch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Wrapper around MLflow for RL experiment tracking."""

    def __init__(self, experiment_name: str = "spot-rl-optimization",
                 tracking_uri: str = "mlruns"):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (local path or server URL)
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run = None
        logger.info(f"MLflow experiment: {experiment_name}, tracking: {tracking_uri}")

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"MLflow run started: {run_name} (id: {self.run.info.run_id})")
        return self.run

    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters (flattens nested dicts)."""
        flat = self._flatten_dict(params, prefix)
        # MLflow has 500 char limit per param value
        for k, v in flat.items():
            mlflow.log_param(k, str(v)[:500])

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics at a given step."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int = 0):
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_model(self, model_path: str, artifact_path: str = "model"):
        """Log model file as artifact."""
        mlflow.log_artifact(model_path, artifact_path)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        """Log any file as artifact."""
        mlflow.log_artifact(path, artifact_path)

    def log_config(self, config: dict):
        """Log config as both params and artifact."""
        self.log_params(config)
        # Also save config.yaml as artifact
        config_path = Path("_tmp_config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        mlflow.log_artifact(str(config_path))
        config_path.unlink()

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if self.run:
            mlflow.end_run(status=status)
            logger.info(f"MLflow run ended: {status}")
            self.run = None

    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        """Flatten nested dict with dot notation."""
        flat = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_dict(v, key))
            else:
                flat[key] = v
        return flat
