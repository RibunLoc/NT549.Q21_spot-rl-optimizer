"""
Register trained models to MLflow Model Registry.

Usage:
    python experiments/register_models.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "mlruns"
REGISTRY_MODEL_NAME = "dqn-spot-optimizer"

MODELS = {
    "stable": {
        "model_path": "results/models/dqn_stable_v5_best.pth",
        "description": "DQN agent trained on stable price scenario. "
                       "Avg Reward: 3177, SLA: 99%, Savings: 40%",
    },
    "volatile": {
        "model_path": "results/models/dqn_volatile_v5_best.pth",
        "description": "DQN agent trained on volatile price scenario. "
                       "Avg Reward: 1994, SLA: 99%, Savings: 30%",
    },
    "spike": {
        "model_path": "results/models/dqn_spike_v5_best.pth",
        "description": "DQN agent trained on workload spike scenario. "
                       "Avg Reward: 2983, SLA: 99%, Savings: 38%",
    },
}


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Use spot-rl-optimization experiment for registration
    mlflow.set_experiment("spot-rl-optimization")

    for scenario, info in MODELS.items():
        model_path = Path(info["model_path"])
        if not model_path.exists():
            print(f"  SKIP {scenario}: {model_path} not found")
            continue

        model_name = f"{REGISTRY_MODEL_NAME}-{scenario}"
        print(f"\n{'='*60}")
        print(f"  Registering: {model_name}")
        print(f"  Model: {model_path}")

        # Start a run to log the model artifact
        with mlflow.start_run(run_name=f"register_{scenario}") as run:
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.set_tag("scenario", scenario)
            mlflow.set_tag("purpose", "model_registry")

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, model_name)

            print(f"  Registered: {model_name} v{result.version}")

        # Update description & transition to Production
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=info["description"],
        )

        # Set alias "champion" (MLflow 2.x uses aliases instead of stages)
        try:
            client.set_registered_model_alias(model_name, "champion", result.version)
            print(f"  Alias: champion -> v{result.version}")
        except Exception:
            # Fallback for older MLflow: use tags
            client.set_model_version_tag(
                model_name, result.version, "stage", "Production"
            )
            print(f"  Tagged as Production: v{result.version}")

        print(f"  Description: {info['description']}")

    # Print summary
    print(f"\n{'='*60}")
    print("  MODEL REGISTRY SUMMARY")
    print(f"{'='*60}")

    for scenario in MODELS:
        model_name = f"{REGISTRY_MODEL_NAME}-{scenario}"
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                print(f"  {model_name} v{v.version} | status: {v.status}")
        except Exception:
            print(f"  {model_name}: not found")

    print(f"\nView in MLflow UI: http://localhost:5000/#/models")
    print("Done!")


if __name__ == "__main__":
    main()
