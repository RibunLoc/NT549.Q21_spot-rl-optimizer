"""Thử kết nối MLflow trên DagsHub TRƯỚC khi tốn GPU Kaggle.

Kiểm tra đúng những gì pipeline dùng: tạo run, log param/metric/artifact,
gắn tag, tìm run theo tag (train-check), register model + alias (eval gate).

Chạy:
    python scripts/mlflow_smoke_test.py      # script tự hỏi token nếu chưa set env
  hoặc set env trước (cùng cửa sổ terminal):
    $env:DAGSHUB_USER_TOKEN = "<token>"      # PowerShell
    export DAGSHUB_USER_TOKEN="<token>"      # Git Bash

Tạo experiment `smoke-test` và registered model `smoke-test-model` trên DagsHub;
xoá trên UI sau khi thử xong.
"""
import os
import sys
import tempfile

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Chạy từ repo root hay scripts/ đều import được experiments.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.mlflow_registry import register_artifact  # noqa: E402  (cùng hàm train dùng)

OWNER = "RibunLoc"
REPO = "NT549.Q21_spot-rl-optimizer"


def step(name, fn):
    try:
        out = fn()
        print(f"[OK]   {name}")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return None


def main() -> int:
    token = os.environ.get("DAGSHUB_USER_TOKEN", "").strip()
    if not token:
        # Không set env thì hỏi trực tiếp (gõ không hiện ra màn hình)
        import getpass
        token = getpass.getpass("DagsHub token (không hiện khi gõ/dán): ").strip()
    if not token:
        print("Không có token DagsHub")
        return 1
    os.environ["MLFLOW_TRACKING_USERNAME"] = OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/{OWNER}/{REPO}.mlflow")
    print("mlflow client:", mlflow.__version__)

    client = MlflowClient()
    failed = 0

    if step("set_experiment", lambda: mlflow.set_experiment("smoke-test")) is None:
        return 1

    with mlflow.start_run(run_name="smoke") as run:
        run_id = run.info.run_id
        failed += step("log_param/metric", lambda: (mlflow.log_param("p", 1),
                                                    mlflow.log_metric("m", 0.5)) or True) is None
        failed += step("set_tags", lambda: mlflow.set_tags({"git_sha": "test", "gate_status": "pending"}) or True) is None

        def _artifact():
            with tempfile.TemporaryDirectory() as d:
                f = os.path.join(d, "model.txt")
                with open(f, "w") as fh:
                    fh.write("dummy model")
                mlflow.log_artifact(f, artifact_path="models")
            return True
        failed += step("log_artifact", _artifact) is None

    def _search():
        exp = client.get_experiment_by_name("smoke-test")
        runs = client.search_runs([exp.experiment_id],
                                  filter_string="tags.gate_status = 'pending' and attributes.status = 'FINISHED'")
        assert any(r.info.run_id == run_id for r in runs), "không tìm thấy run theo tag"
        return True
    failed += step("search_runs theo tag (train-check)", _search) is None

    mv = step("register_artifact (giống train_generalist)",
              lambda: register_artifact(client, "smoke-test-model", run_id=run_id,
                                        artifact_path="models/model.txt", tags={"avg_cost": "1.0"}))
    failed += mv is None
    if mv is not None:
        failed += step("set alias @champion (eval gate)",
                       lambda: client.set_registered_model_alias("smoke-test-model", "champion", mv.version) or True) is None
        failed += step("get_model_version_download_uri",
                       lambda: print("       →", client.get_model_version_download_uri("smoke-test-model", mv.version)) or True) is None

    print("\nKẾT QUẢ:", "tất cả OK" if not failed else f"{failed} bước lỗi")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
