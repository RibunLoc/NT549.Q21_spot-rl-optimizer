"""Đăng ký artifact thường (.pth / .onnx) vào MLflow Model Registry.

Vì sao không dùng `mlflow.register_model("runs:/<id>/<path>")`:
MLflow client 3.x coi URI đó là một *logged model* (tạo bằng
`mlflow.<flavor>.log_model`). File log bằng `log_artifact` không phải logged model
→ lỗi "Unable to find a logged_model with artifact_path ...".
Ở đây gọi thẳng `create_model_version` với `source` là đường dẫn artifact thật
của run — chạy được với cả MLflow 2.x lẫn 3.x, trên DagsHub hay server tự host.
"""
from typing import Dict, Optional

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def register_artifact(
    client: MlflowClient,
    name: str,
    run_id: str,
    artifact_path: str,
    tags: Optional[Dict[str, str]] = None,
) -> ModelVersion:
    """Tạo version mới của registered model `name` trỏ tới `<artifact_uri của run>/<artifact_path>`."""
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)

    artifact_uri = client.get_run(run_id).info.artifact_uri.rstrip("/")
    source = f"{artifact_uri}/{artifact_path.lstrip('/')}"
    return client.create_model_version(
        name=name,
        source=source,
        run_id=run_id,
        tags={k: str(v) for k, v in (tags or {}).items()},
    )
