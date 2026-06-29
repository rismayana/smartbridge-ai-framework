"""
SHMS — MLflow Model Registry Helper
Mendaftarkan, mempromosikan, dan memuat model dari MLflow Registry (DagsHub).

Alur model registry:
  Training run  →  Staging  →  Production
                              (model terbaik yang sudah divalidasi)

Nama model di registry:
  shms-isolation-forest    (Phase 3B, lokal)
  shms-lstm-autoencoder    (Phase 3A, Colab)
  shms-gnn-autoencoder     (Phase 3C, Colab)
  shms-ensemble            (Phase 4, lokal)

Cara pakai:
  # Daftarkan model dari run terakhir IForest
  from model_registry import register_best_iforest
  register_best_iforest()

  # Promosikan ke Production
  from model_registry import promote_model
  promote_model("shms-isolation-forest", version=1, stage="Production")

  # Load model Production untuk inference
  from model_registry import load_production_model
  model = load_production_model("shms-isolation-forest")
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Nama model standar di registry ───────────────────────────────
MODEL_NAMES = {
    "iforest"  : "shms-isolation-forest",
    "lstm"     : "shms-lstm-autoencoder",
    "gnn"      : "shms-gnn-autoencoder",
    "ensemble" : "shms-ensemble",
}

VALID_STAGES = ("Staging", "Production", "Archived", "None")


def _get_client():
    """Return MlflowClient dengan tracking URI yang sudah dikonfigurasi."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow_config import setup_mlflow
    setup_mlflow()
    return MlflowClient()


# ────────────────────────────────────────────────────────────────
# REGISTER
# ────────────────────────────────────────────────────────────────

def register_model(run_id: str,
                   model_key: str,
                   artifact_path: str = "",
                   stage: str = "Staging") -> object:
    """
    Daftarkan model dari MLflow run ke Model Registry.

    Args:
        run_id       : ID run MLflow (dari mlflow.active_run().info.run_id)
        model_key    : "iforest" | "lstm" | "gnn" | "ensemble"
        artifact_path: path artifact dalam run (default = model_key)
        stage        : stage awal, default "Staging"

    Returns:
        ModelVersion object
    """
    import mlflow
    from mlflow_config import setup_mlflow
    setup_mlflow()

    model_name    = MODEL_NAMES.get(model_key, model_key)
    artifact_path = artifact_path or model_key
    model_uri     = f"runs:/{run_id}/{artifact_path}"

    print(f"\n  [Registry] Mendaftarkan: {model_name}")
    print(f"             URI  : {model_uri}")

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"             Ver  : {mv.version} | Stage: {mv.current_stage}")

    if stage and stage != "None":
        client = _get_client()
        client.transition_model_version_stage(
            name    = model_name,
            version = mv.version,
            stage   = stage,
        )
        print(f"             -> dipromosikan ke: {stage}")

    return mv


def register_best_iforest(stage: str = "Staging") -> object | None:
    """
    Cari run IForest terbaik (F1 tertinggi) dan daftarkan ke registry.
    Berguna untuk auto-register setelah eksperimen selesai.
    """
    import mlflow
    from mlflow_config import setup_mlflow
    setup_mlflow()

    experiment = mlflow.get_experiment_by_name("shms-anomaly-detection")
    if experiment is None:
        print("  [Registry] Experiment belum ada. Jalankan training dulu.")
        return None

    runs = mlflow.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string  = "tags.mlflow.runName = 'isolation_forest'",
        order_by       = ["metrics.f1 DESC"],
        max_results    = 1,
    )

    if runs.empty:
        print("  [Registry] Tidak ada run isolation_forest ditemukan.")
        return None

    best_run = runs.iloc[0]
    run_id   = best_run["run_id"]
    f1       = best_run.get("metrics.f1", 0)
    print(f"  [Registry] Run terbaik: {run_id} | F1={f1:.4f}")

    return register_model(run_id, "iforest", "isolation_forest", stage)


# ────────────────────────────────────────────────────────────────
# PROMOTE
# ────────────────────────────────────────────────────────────────

def promote_model(model_key: str, version: int, stage: str = "Production"):
    """
    Promosikan versi model ke stage tertentu.

    Args:
        model_key : "iforest" | "lstm" | "gnn" | "ensemble"
        version   : nomor versi (int)
        stage     : "Staging" | "Production" | "Archived"
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Stage harus salah satu dari: {VALID_STAGES}")

    client     = _get_client()
    model_name = MODEL_NAMES.get(model_key, model_key)

    client.transition_model_version_stage(
        name    = model_name,
        version = version,
        stage   = stage,
    )
    print(f"  [Registry] {model_name} v{version} -> {stage}")


# ────────────────────────────────────────────────────────────────
# LOAD
# ────────────────────────────────────────────────────────────────

def load_production_model(model_key: str):
    """
    Load model versi Production dari registry untuk inference.

    Args:
        model_key: "iforest" | "lstm" | "gnn" | "ensemble"

    Returns:
        model object (sklearn / pytorch tergantung model)
    """
    import mlflow.sklearn
    import mlflow.pytorch
    from mlflow_config import setup_mlflow
    setup_mlflow()

    model_name = MODEL_NAMES.get(model_key, model_key)
    model_uri  = f"models:/{model_name}/Production"

    print(f"  [Registry] Loading: {model_uri}")

    # Coba sklearn dulu, fallback ke pytorch
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        model = mlflow.pytorch.load_model(model_uri)

    print(f"  [Registry] Model loaded: {type(model).__name__}")
    return model


# ────────────────────────────────────────────────────────────────
# LIST & INFO
# ────────────────────────────────────────────────────────────────

def list_models(verbose: bool = True) -> list:
    """List semua model terdaftar beserta versi dan stage-nya."""
    client = _get_client()
    models = client.search_registered_models()

    if not models:
        print("  [Registry] Belum ada model terdaftar.")
        return []

    if verbose:
        print(f"\n  {'Model':<35} {'Ver':>4}  {'Stage':<12}  {'F1':>6}")
        print(f"  {'-'*35} {'-'*4}  {'-'*12}  {'-'*6}")
        for rm in models:
            for mv in client.get_latest_versions(rm.name):
                # Ambil metric F1 dari run asal jika ada
                try:
                    import mlflow
                    run = mlflow.get_run(mv.run_id)
                    f1  = run.data.metrics.get("f1", float("nan"))
                    f1_str = f"{f1:.4f}" if f1 == f1 else "  -   "
                except Exception:
                    f1_str = "  -   "
                print(f"  {rm.name:<35} {mv.version:>4}  {mv.current_stage:<12}  {f1_str}")
        print()

    return models


def get_model_info(model_key: str) -> dict:
    """Ambil info lengkap model dari registry (semua versi + stage)."""
    client     = _get_client()
    model_name = MODEL_NAMES.get(model_key, model_key)

    try:
        rm       = client.get_registered_model(model_name)
        versions = client.get_latest_versions(model_name)
        return {
            "name"        : rm.name,
            "description" : rm.description,
            "versions"    : [
                {
                    "version" : mv.version,
                    "stage"   : mv.current_stage,
                    "run_id"  : mv.run_id,
                    "created" : mv.creation_timestamp,
                }
                for mv in versions
            ],
        }
    except Exception as e:
        print(f"  [Registry] Model '{model_name}' tidak ditemukan: {e}")
        return {}


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHMS MLflow Model Registry")
    sub    = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List semua model terdaftar")

    p_reg = sub.add_parser("register", help="Daftarkan model terbaik IForest")
    p_reg.add_argument("--stage", default="Staging",
                       choices=["Staging", "Production"])

    p_pro = sub.add_parser("promote", help="Promosikan model ke stage tertentu")
    p_pro.add_argument("model",   help="iforest | lstm | gnn | ensemble")
    p_pro.add_argument("version", type=int, help="Nomor versi")
    p_pro.add_argument("--stage", default="Production",
                       choices=["Staging", "Production", "Archived"])

    p_load = sub.add_parser("load", help="Test load model dari Production")
    p_load.add_argument("model", help="iforest | lstm | gnn | ensemble")

    args = parser.parse_args()

    if args.cmd == "list":
        list_models()
    elif args.cmd == "register":
        register_best_iforest(stage=args.stage)
    elif args.cmd == "promote":
        promote_model(args.model, args.version, args.stage)
    elif args.cmd == "load":
        load_production_model(args.model)
    else:
        parser.print_help()
