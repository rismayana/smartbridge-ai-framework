"""
SHMS — Config Loader
Load hyperparameter dan konfigurasi dari config.yaml di root project.

Cara pakai:
  from config_loader import load_config, get_hp

  cfg        = load_config()
  hp_iforest = get_hp("iforest")   # dict HP Isolation Forest
  hp_lstm    = get_hp("lstm")      # dict HP LSTM
  hp_gnn     = get_hp("gnn")       # dict HP GNN

Fallback:
  Jika config.yaml tidak ditemukan, dikembalikan dict kosong {}
  sehingga script tetap bisa berjalan dengan HP default bawaan.
"""

import warnings
from pathlib import Path
from functools import lru_cache

# Cari config.yaml dari lokasi file ini (03_code/) naik ke root
_CONFIG_CANDIDATES = [
    Path(__file__).parent.parent / "config.yaml",   # root project (utama)
    Path(__file__).parent / "config.yaml",           # 03_code/ (fallback)
]

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    Load config.yaml dan return sebagai dict.
    Di-cache setelah pertama kali dipanggil (tidak baca ulang tiap call).
    """
    if not _YAML_AVAILABLE:
        warnings.warn(
            "[config_loader] PyYAML tidak terinstall. Jalankan: pip install pyyaml\n"
            "Menggunakan HP default dari masing-masing script.",
            stacklevel=2
        )
        return {}

    for path in _CONFIG_CANDIDATES:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            print(f"  [config] Loaded: {path}")
            return cfg

    warnings.warn(
        "[config_loader] config.yaml tidak ditemukan. "
        "Menggunakan HP default dari masing-masing script.",
        stacklevel=2
    )
    return {}


def get_hp(model: str) -> dict:
    """
    Ambil HP untuk model tertentu dari config.yaml.

    Args:
        model: "iforest" | "lstm" | "gnn" | "ensemble"

    Returns:
        dict HP, atau {} jika model/config tidak ditemukan
    """
    cfg = load_config()
    return cfg.get("models", {}).get(model, {})


def get_data_config() -> dict:
    """Ambil konfigurasi data/preprocessing."""
    return load_config().get("data", {})


def get_mlops_config() -> dict:
    """Ambil konfigurasi MLOps (tracking tool, experiment name, dll)."""
    return load_config().get("mlops", {})


if __name__ == "__main__":
    cfg = load_config()
    if cfg:
        print(f"\n  Project : {cfg.get('project', {}).get('name')}")
        print(f"  MLOps   : {cfg.get('mlops', {}).get('tracking')}")
        print(f"  Models  : {list(cfg.get('models', {}).keys())}")
    else:
        print("  config.yaml tidak ditemukan atau kosong.")
