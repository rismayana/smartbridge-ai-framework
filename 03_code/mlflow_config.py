"""
SHMS — MLflow Configuration Helper
Mendukung DagsHub sebagai tracking server (lokal & Google Colab).

Setup awal (sekali saja):
  1. Daftar di https://dagshub.com → connect repo GitHub shms-ai-anomaly-detection
  2. Buat file .env di root project:
       DAGSHUB_TOKEN=<token_dari_dagshub_settings>
  3. Semua eksperimen (lokal & Colab) otomatis terekam di DagsHub MLflow UI.

Cara pakai di script:
  from mlflow_config import setup_mlflow
  setup_mlflow("shms-anomaly-detection")

Cara pakai di Colab:
  Lihat file: 03_code/notebooks/mlflow_colab_snippet.py
"""

import os
import mlflow
from pathlib import Path

DAGSHUB_USERNAME    = "rismayana"
DAGSHUB_REPO        = "shms-ai-anomaly-detection"
MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)


def _load_token() -> str | None:
    """Cari DAGSHUB_TOKEN dari env var atau file .env."""
    token = os.environ.get("DAGSHUB_TOKEN")
    if token:
        return token

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DAGSHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def setup_mlflow(experiment_name: str = "") -> str:
    """
    Setup MLflow tracking URI.
    - Jika DAGSHUB_TOKEN tersedia → tracking ke DagsHub (cloud, bisa dari Colab)
    - Jika tidak → fallback ke lokal (mlruns/ di root project)

    Returns:
        tracking URI yang aktif
    """
    # Prioritas experiment_name: argumen → config.yaml → default
    if not experiment_name:
        try:
            from config_loader import get_mlops_config
            experiment_name = get_mlops_config().get(
                "experiment_name", "shms-anomaly-detection"
            )
        except Exception:
            experiment_name = "shms-anomaly-detection"

    token = _load_token()

    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"  [MLflow] DagsHub  : {MLFLOW_TRACKING_URI}")
    else:
        local_uri = Path(__file__).parent.parent / "mlruns"
        mlflow.set_tracking_uri(local_uri.as_uri())
        print(f"  [MLflow] Lokal    : {local_uri}")
        print(f"  [INFO]   Set DAGSHUB_TOKEN di .env untuk sync ke DagsHub")

    mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()


def log_dataset_info(split: str, n_windows: int, n_abnormal: int):
    """Helper: log info dataset sebagai MLflow tags."""
    mlflow.set_tag(f"data.{split}.n_windows",  n_windows)
    mlflow.set_tag(f"data.{split}.n_abnormal", n_abnormal)
    mlflow.set_tag(f"data.{split}.pct_abnormal",
                   round(100 * n_abnormal / max(n_windows, 1), 2))


if __name__ == "__main__":
    uri = setup_mlflow()
    print(f"\n  Tracking URI aktif: {uri}")
    print(f"  Untuk melihat UI lokal: mlflow ui --backend-store-uri {uri}")
