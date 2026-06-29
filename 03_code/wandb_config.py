"""
SHMS — Weights & Biases (W&B) Configuration Helper
Mendukung tracking dari lokal maupun Google Colab tanpa server sendiri.

Setup awal (sekali saja):
  1. Daftar di https://wandb.ai (gratis untuk research)
  2. Buat file .env di root project:
       WANDB_API_KEY=<api_key_dari_wandb_settings>
  3. Semua eksperimen (lokal & Colab) otomatis terekam di wandb.ai dashboard.

Cara pakai di script:
  from wandb_config import setup_wandb, finish_wandb
  run = setup_wandb(run_name="isolation_forest", config=HP)
  # ... training ...
  finish_wandb()

Cara pakai di Colab:
  Lihat file: 03_code/notebooks/wandb_colab_snippet.py
"""

import os
import sys
from pathlib import Path

WANDB_PROJECT = "shms-anomaly-detection"
WANDB_ENTITY  = "rismayana"          # username W&B


def _load_api_key() -> str | None:
    """Cari WANDB_API_KEY dari env var atau file .env."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("WANDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def setup_wandb(run_name: str, config: dict,
                tags: list[str] | None = None,
                notes: str = ""):
    """
    Inisialisasi W&B run.
    - Jika WANDB_API_KEY tersedia → log ke wandb.ai (cloud)
    - Jika tidak → mode offline (log ke disk, bisa di-sync nanti)

    Args:
        run_name : nama run, misal "isolation_forest", "lstm_autoencoder"
        config   : dict hyperparameter
        tags     : list tag, misal ["phase3b", "iforest", "cpu"]
        notes    : deskripsi singkat run

    Returns:
        wandb.run object, atau None jika wandb tidak terinstall
    """
    try:
        import wandb
    except ImportError:
        print("  [W&B] wandb tidak terinstall. Jalankan: pip install wandb")
        return None

    api_key = _load_api_key()

    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        mode = "online"
        print(f"  [W&B] Online  : https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
    else:
        mode = "offline"
        print(f"  [W&B] Offline : hasil disimpan lokal, sync nanti dengan 'wandb sync'")
        print(f"  [INFO] Set WANDB_API_KEY di .env untuk sync otomatis ke wandb.ai")

    run = wandb.init(
        project = WANDB_PROJECT,
        entity  = WANDB_ENTITY,
        name    = run_name,
        config  = config,
        tags    = tags or [],
        notes   = notes,
        mode    = mode,
        reinit  = True,
    )
    return run


def finish_wandb():
    """Tutup W&B run dengan aman."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    run = setup_wandb(
        run_name = "test_connection",
        config   = {"test": True},
        tags     = ["test"],
        notes    = "Cek koneksi W&B"
    )
    if run:
        import wandb
        wandb.log({"dummy_metric": 1.0})
        finish_wandb()
        print("  Koneksi W&B berhasil.")
