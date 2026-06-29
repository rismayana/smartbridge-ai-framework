"""
SHMS Bridge Anomaly Detection
shms_phase3b_iforest.py — Isolation Forest

Prinsip kerja:
  - Algoritma berbasis pohon keputusan (tree-based)
  - Anomali lebih mudah "diisolasi" dari data normal
  - Tidak butuh asumsi distribusi data (non-parametric)
  - Jauh lebih ringan dari LSTM — bisa jalan di CPU biasa
  - Input: fitur statistik per window (bukan raw time-series)

Fitur yang digunakan per channel (8 fitur × 17 channel = 136 fitur):
  mean, std, min, max, peak-to-peak, RMS, skewness, kurtosis

Keunggulan sebagai ensemble member:
  - Mendeteksi anomali dari perspektif STATISTIK (bukan temporal)
  - Melengkapi LSTM yang fokus pada pola temporal
  - Sangat cepat — inference < 1 detik untuk ribuan windows

Cara pakai:
  python shms_phase3b_iforest.py --train
  python shms_phase3b_iforest.py --eval
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from shms_config import (
    MODEL_DIR, RESULTS_DIR, DATA_PROCESSED_DIR,
    MAIN_CHANNELS, CHANNEL_ALIAS, WINDOW_SIZE,
)


# ─────────────────────────────────────────────────────────
# HYPERPARAMETER
# ─────────────────────────────────────────────────────────

HP = {
    "n_estimators"      : 200,    # jumlah pohon — lebih banyak lebih stabil
    "max_samples"       : 0.8,    # proporsi sampel per pohon
    "contamination"     : 0.05,   # estimasi proporsi anomali (5%)
    "max_features"      : 1.0,    # proporsi fitur per pohon
    "random_state"      : 42,
    "n_jobs"            : -1,     # pakai semua CPU core
    "threshold_pct"     : 95,     # persentil score untuk threshold
}


# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────

FEATURE_NAMES_PER_CH = [
    "mean", "std", "min", "max",
    "ptp",          # peak-to-peak (max - min)
    "rms",          # root mean square
    "skewness",     # kemiringan distribusi
    "kurtosis",     # ketajaman distribusi
]

def extract_features(X: np.ndarray,
                     channel_names: list) -> pd.DataFrame:
    """
    Ekstrak 8 fitur statistik dari setiap window dan channel.

    Args:
        X             : (n_windows, window_size, n_channels)
        channel_names : nama channel (untuk nama kolom)

    Returns:
        DataFrame (n_windows, n_channels × 8_features)
    """
    n_windows, win_size, n_ch = X.shape
    records = []

    for i in range(n_windows):
        row = {}
        for j, ch in enumerate(channel_names):
            alias = CHANNEL_ALIAS.get(ch, ch.replace("FB_", ""))
            s = X[i, :, j]
            s_clean = s[~np.isnan(s)]

            if len(s_clean) < 2:
                for feat in FEATURE_NAMES_PER_CH:
                    row[f"{alias}_{feat}"] = 0.0
                continue

            row[f"{alias}_mean"]     = float(s_clean.mean())
            row[f"{alias}_std"]      = float(s_clean.std())
            row[f"{alias}_min"]      = float(s_clean.min())
            row[f"{alias}_max"]      = float(s_clean.max())
            row[f"{alias}_ptp"]      = float(s_clean.max() - s_clean.min())
            row[f"{alias}_rms"]      = float(np.sqrt((s_clean**2).mean()))
            row[f"{alias}_skewness"] = float(stats.skew(s_clean))
            row[f"{alias}_kurtosis"] = float(stats.kurtosis(s_clean))

        records.append(row)

    feat_df = pd.DataFrame(records)
    return feat_df


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────

def load_split(split: str, normal_only: bool = False,
               max_windows: int = None) -> tuple:
    """
    Load data .npy. Fallback ke simulasi jika belum ada.
    """
    summary_path = DATA_PROCESSED_DIR / "processing_summary.csv"

    if not summary_path.exists():
        print(f"  [SIMULASI] Membuat data dummy untuk '{split}'...")
        n  = {"train": 5000, "val": 1000, "test": 1000}[split]
        X  = np.random.randn(n, WINDOW_SIZE,
                              len(MAIN_CHANNELS)).astype(np.float32)
        y  = np.zeros(n, dtype=np.int8)
        if split != "train":
            abn = np.random.choice(n, n // 10, replace=False)
            X[abn] += (np.random.randn(len(abn), WINDOW_SIZE,
                                        len(MAIN_CHANNELS)) * 3).astype(np.float32)
            y[abn] = 1
        return X, y

    summary = pd.read_csv(summary_path)
    days    = summary[
        (summary["split"] == split) & (summary["status"] == "ok")
    ]["date"].tolist()

    X_list, y_list = [], []
    for d in sorted(days):
        xp = DATA_PROCESSED_DIR / f"{d}_X.npy"
        yp = DATA_PROCESSED_DIR / f"{d}_y.npy"
        if xp.exists():
            X_list.append(np.load(xp))
            y_list.append(np.load(yp))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)

    if normal_only:
        X, y = X[y == 0], y[y == 0]

    if max_windows and len(X) > max_windows:
        idx = np.sort(np.random.choice(len(X), max_windows, replace=False))
        X, y = X[idx], y[idx]

    print(f"  [{split}] {len(X):,} windows "
          f"({y.sum()} abnormal, {100*y.mean():.1f}%)")
    return X, y


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────

def train(hp: dict = HP,
          save_dir: Path = MODEL_DIR,
          results_dir: Path = RESULTS_DIR) -> tuple:
    """
    Training Isolation Forest.
    1. Load data train (normal only)
    2. Ekstrak fitur statistik
    3. Fit StandardScaler + IsolationForest
    4. Simpan model + scaler
    """
    print(f"\n{'='*65}")
    print(f"  ISOLATION FOREST — TRAINING")
    print(f"  n_estimators={hp['n_estimators']} | "
          f"contamination={hp['contamination']} | "
          f"max_samples={hp['max_samples']}")
    print(f"{'='*65}")

    # Load data
    print("\n[DATA]")
    X_train, _ = load_split("train", normal_only=True)

    # Ekstrak fitur
    print("\n[FEATURE EXTRACTION]")
    print(f"  Windows: {len(X_train):,} | "
          f"Channels: {X_train.shape[2]} | "
          f"Fitur per channel: {len(FEATURE_NAMES_PER_CH)}")
    feat_train = extract_features(X_train, MAIN_CHANNELS)
    n_feat = feat_train.shape[1]
    print(f"  Total fitur: {n_feat} "
          f"({X_train.shape[2]} channels × {len(FEATURE_NAMES_PER_CH)} fitur)")

    # Simpan nama fitur
    feat_path = save_dir / "iforest_feature_names.json"
    save_dir.mkdir(exist_ok=True)
    with open(feat_path, "w") as f:
        json.dump(feat_train.columns.tolist(), f)

    # Scaling — IsolationForest tidak wajib scaling,
    # tapi membantu stabilitas jika skala channel berbeda jauh
    print("\n[SCALING]")
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_train.fillna(0))
    print(f"  Scaler fitted pada {feat_scaled.shape[0]:,} samples")

    # Training Isolation Forest
    print("\n[TRAINING]")
    iforest = IsolationForest(
        n_estimators  = hp["n_estimators"],
        max_samples   = hp["max_samples"],
        contamination = hp["contamination"],
        max_features  = hp["max_features"],
        random_state  = hp["random_state"],
        n_jobs        = hp["n_jobs"],
        verbose       = 0,
    )
    iforest.fit(feat_scaled)
    print(f"  Isolation Forest fitted ✓")

    # Simpan model & scaler
    joblib.dump(iforest, save_dir / "isolation_forest.pkl")
    joblib.dump(scaler,  save_dir / "iforest_scaler.pkl")

    # Simpan HP
    with open(save_dir / "iforest_hp.json", "w") as f:
        json.dump(hp, f, indent=2)

    print(f"\n  Model disimpan: {save_dir / 'isolation_forest.pkl'}")

    # Kalibrasi threshold dari validation set
    threshold = calibrate_threshold(iforest, scaler, hp, save_dir, results_dir)

    return iforest, scaler, threshold


# ─────────────────────────────────────────────────────────
# THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────

def calibrate_threshold(iforest, scaler, hp: dict = HP,
                        save_dir: Path = MODEL_DIR,
                        results_dir: Path = RESULTS_DIR) -> float:
    """
    Kalibrasi threshold dari anomaly score pada data validation.

    IsolationForest.score_samples() → skor negatif:
      Semakin negatif = semakin anomali
      Kita balikkan tanda → semakin positif = semakin anomali
    """
    print("\n[THRESHOLD CALIBRATION]")
    X_val, y_val = load_split("val", normal_only=False)

    feat_val    = extract_features(X_val, MAIN_CHANNELS)
    feat_scaled = scaler.transform(feat_val.fillna(0))

    # score_samples: lebih rendah = lebih anomali
    # Balikkan tanda agar lebih tinggi = lebih anomali
    raw_scores  = -iforest.score_samples(feat_scaled)

    scores_normal = raw_scores[y_val == 0]
    threshold     = np.percentile(scores_normal, hp["threshold_pct"])

    print(f"  Skor normal  : mean={scores_normal.mean():.4f}, "
          f"std={scores_normal.std():.4f}")
    print(f"  Threshold    : {threshold:.4f} "
          f"(persentil {hp['threshold_pct']})")

    # Simpan threshold
    with open(save_dir / "iforest_threshold.json", "w") as f:
        json.dump({
            "threshold"         : float(threshold),
            "threshold_pct"     : hp["threshold_pct"],
            "score_mean_normal" : float(scores_normal.mean()),
            "score_std_normal"  : float(scores_normal.std()),
        }, f, indent=2)

    # Plot distribusi skor
    _plot_score_distribution(
        raw_scores, y_val, threshold,
        results_dir / "figures" / "iforest_score_distribution.png"
    )

    return threshold


# ─────────────────────────────────────────────────────────
# EVALUASI
# ─────────────────────────────────────────────────────────

def evaluate(iforest=None, scaler=None, threshold: float = None,
             save_dir: Path = MODEL_DIR,
             results_dir: Path = RESULTS_DIR) -> dict:
    """Evaluasi pada data test."""

    if iforest is None or scaler is None:
        iforest, scaler, threshold = load_model(save_dir)

    if threshold is None:
        with open(save_dir / "iforest_threshold.json") as f:
            threshold = json.load(f)["threshold"]

    print(f"\n[EVALUASI] Data test | Threshold: {threshold:.4f}")

    X_test, y_test = load_split("test", normal_only=False)

    feat_test   = extract_features(X_test, MAIN_CHANNELS)
    feat_scaled = scaler.transform(feat_test.fillna(0))
    raw_scores  = -iforest.score_samples(feat_scaled)

    # Normalisasi ke [0,1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    scores_norm  = (raw_scores - s_min) / (s_max - s_min + 1e-9)

    y_pred     = (raw_scores > threshold).astype(int)
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    auc        = roc_auc_score(y_test, scores_norm) if y_test.sum() > 0 else 0.0
    cm         = confusion_matrix(y_test, y_pred)

    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:6,}  FP={cm[0,1]:6,}")
    print(f"    FN={cm[1,0]:6,}  TP={cm[1,1]:6,}")
    print(f"\n  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    metrics = {
        "model"     : "Isolation_Forest",
        "threshold" : threshold,
        "precision" : round(precision, 4),
        "recall"    : round(recall,    4),
        "f1"        : round(f1,        4),
        "auc"       : round(auc,       4),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        "n_test_windows"   : len(y_test),
        "n_test_abnormal"  : int(y_test.sum()),
        "n_features"       : feat_test.shape[1],
    }

    results_dir.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        results_dir / "iforest_metrics.csv", index=False
    )

    # Plots
    _plot_evaluation(
        raw_scores, scores_norm, y_test, y_pred, threshold, cm,
        results_dir / "figures"
    )

    # Feature importance
    _plot_feature_importance(
        iforest, feat_test.columns.tolist(),
        results_dir / "figures" / "iforest_feature_importance.png"
    )

    return metrics


# ─────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────

def predict(X: np.ndarray, iforest=None, scaler=None,
            threshold: float = None,
            save_dir: Path = MODEL_DIR) -> dict:
    """
    Deteksi anomali pada data baru.
    Input X: (n_windows, WINDOW_SIZE, n_channels) — sudah dinormalisasi
    """
    if iforest is None:
        iforest, scaler, threshold = load_model(save_dir)

    feat        = extract_features(X, MAIN_CHANNELS)
    feat_scaled = scaler.transform(feat.fillna(0))
    raw_scores  = -iforest.score_samples(feat_scaled)

    s_min, s_max = raw_scores.min(), raw_scores.max()
    scores_norm  = (raw_scores - s_min) / (s_max - s_min + 1e-9)
    preds        = (raw_scores > threshold).astype(int)

    return {
        "raw_scores" : raw_scores,
        "scores"     : scores_norm,
        "predictions": preds,
        "threshold"  : threshold,
    }


def load_model(save_dir: Path = MODEL_DIR):
    """Load model, scaler, dan threshold dari file tersimpan."""
    model_path = save_dir / "isolation_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    iforest   = joblib.load(save_dir / "isolation_forest.pkl")
    scaler    = joblib.load(save_dir / "iforest_scaler.pkl")
    threshold = None

    thresh_path = save_dir / "iforest_threshold.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            threshold = json.load(f)["threshold"]

    print(f"  Model loaded: {model_path.name} | threshold={threshold}")
    return iforest, scaler, threshold


# ─────────────────────────────────────────────────────────
# HELPER PLOTS
# ─────────────────────────────────────────────────────────

def _plot_score_distribution(scores, y_true, threshold, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sc_normal = scores[y_true == 0]
    sc_abnorm = scores[y_true == 1] if y_true.sum() > 0 else np.array([])

    axes[0].hist(sc_normal, bins=60, alpha=0.75, color="#378ADD", label="Normal")
    if len(sc_abnorm):
        axes[0].hist(sc_abnorm, bins=40, alpha=0.75, color="#E24B4A", label="Abnormal")
    axes[0].axvline(threshold, color="#BA7517", lw=2, linestyle="--",
                    label=f"Threshold={threshold:.4f}")
    axes[0].set_xlabel("Anomaly Score (lebih tinggi = lebih anomali)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribusi Score — Validation Set")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    pcts = np.arange(50, 100, 0.5)
    vals = [np.percentile(sc_normal, p) for p in pcts]
    axes[1].plot(pcts, vals, color="#378ADD", lw=1.8)
    axes[1].axvline(95, color="#BA7517", lw=1.5, linestyle="--", label="P95 (default)")
    axes[1].set_xlabel("Persentil")
    axes[1].set_ylabel("Score value")
    axes[1].set_title("Kurva Persentil Score Normal")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Isolation Forest — Threshold Calibration", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path.name}")


def _plot_evaluation(raw_scores, scores_norm, y_test, y_pred,
                     threshold, cm, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(["Normal","Abnormal"])
    axes[0].set_yticklabels(["Normal","Abnormal"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix — Isolation Forest")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if cm[i,j] > cm.max()//2 else "black")
    plt.colorbar(im, ax=axes[0])

    if y_test.sum() > 0:
        from sklearn.metrics import roc_curve as _roc
        fpr, tpr, _ = _roc(y_test, scores_norm)
        auc = roc_auc_score(y_test, scores_norm)
        axes[1].plot(fpr, tpr, color="#378ADD", lw=2, label=f"AUC={auc:.4f}")
        axes[1].plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Isolation Forest — Evaluasi Data Test", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir / "iforest_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Timeline
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    idx = np.arange(len(raw_scores))
    axes[0].plot(idx, raw_scores, color="#888780", lw=0.6, alpha=0.8)
    axes[0].axhline(threshold, color="#BA7517", lw=1.5, linestyle="--",
                    label=f"Threshold={threshold:.4f}")
    if y_test.sum():
        abn = np.where(y_test == 1)[0]
        axes[0].scatter(abn, raw_scores[abn], color="#E24B4A",
                        s=6, zorder=5, label="Actual abnormal")
    fp  = np.where((y_pred == 1) & (y_test == 0))[0]
    if len(fp):
        axes[0].scatter(fp, raw_scores[fp], color="#FAC775",
                        s=4, zorder=4, label="False positive")
    axes[0].set_ylabel("Anomaly Score")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    axes[1].fill_between(idx, scores_norm, alpha=0.6, color="#1D9E75")
    axes[1].axhline(0.5, color="#BA7517", lw=1, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Normalized Score (0–1)")
    axes[1].set_xlabel("Window index")
    axes[1].grid(True, alpha=0.2)
    fig.suptitle("Isolation Forest — Anomaly Detection Timeline", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir / "iforest_anomaly_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plots disimpan di: {save_dir}/")


def _plot_feature_importance(iforest, feature_names: list, save_path: Path,
                              top_n: int = 20):
    """
    Estimasi feature importance dari Isolation Forest
    menggunakan rata-rata kedalaman fitur di setiap pohon.
    Fitur dengan rata-rata kedalaman rendah = lebih "informatif" untuk isolasi.
    """
    try:
        # Hitung mean depth per fitur
        importances = np.zeros(len(feature_names))
        for tree in iforest.estimators_:
            feat_used = tree.tree_.feature
            depth     = np.zeros(tree.tree_.n_node_samples.shape)
            # Hitung depth setiap node
            stack = [(0, 0)]
            while stack:
                node, d = stack.pop()
                depth[node] = d
                if tree.tree_.children_left[node] != -1:
                    stack.append((tree.tree_.children_left[node],  d+1))
                    stack.append((tree.tree_.children_right[node], d+1))
            # Akumulasi: fitur yang sering digunakan di level dangkal = penting
            for node in range(tree.tree_.node_count):
                f = feat_used[node]
                if f >= 0 and f < len(feature_names):
                    importances[f] += 1.0 / (depth[node] + 1)

        importances = importances / importances.sum()

        # Ambil top N
        top_idx  = np.argsort(importances)[-top_n:][::-1]
        top_imp  = importances[top_idx]
        top_names = [feature_names[i] for i in top_idx]

        # Warnai berdasarkan tipe sensor
        colors = []
        for n in top_names:
            if n.startswith("AC_"):  colors.append("#378ADD")
            elif n.startswith("CA_"): colors.append("#1D9E75")
            else:                     colors.append("#888780")

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.3)))
        bars = ax.barh(range(top_n), top_imp[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Relative Importance")
        ax.set_title(f"Isolation Forest — Top {top_n} Feature Importance\n"
                     "(Biru=Accelerometer, Hijau=Cable, Abu=lainnya)")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {save_path.name}")

    except Exception as e:
        print(f"  [WARN] Feature importance plot gagal: {e}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def run_phase3b(retrain: bool = False) -> dict:
    """Jalankan Phase 3B lengkap."""
    print(f"\n{'='*65}")
    print(f"  PHASE 3B — ISOLATION FOREST")
    print(f"{'='*65}")

    model_path = MODEL_DIR / "isolation_forest.pkl"

    if not model_path.exists() or retrain:
        iforest, scaler, threshold = train(HP, MODEL_DIR, RESULTS_DIR)
    else:
        print(f"  Model sudah ada, load dari: {model_path}")
        iforest, scaler, threshold = load_model(MODEL_DIR)

    metrics = evaluate(iforest, scaler, threshold, MODEL_DIR, RESULTS_DIR)

    print(f"\n  PHASE 3B SELESAI")
    print(f"  F1={metrics.get('f1', 0):.4f} | "
          f"AUC={metrics.get('auc', 0):.4f}")

    _log_to_mlflow(metrics)

    return metrics


def _log_to_mlflow(metrics: dict):
    """Log hasil ke MLflow jika tersedia. Gagal diam-diam agar tidak ganggu pipeline."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from mlflow_config import setup_mlflow
        setup_mlflow("shms-anomaly-detection")

        with mlflow.start_run(run_name="isolation_forest"):
            # Hyperparameter
            mlflow.log_params({
                "model"          : "IsolationForest",
                "n_estimators"   : HP["n_estimators"],
                "max_samples"    : HP["max_samples"],
                "contamination"  : HP["contamination"],
                "max_features"   : HP["max_features"],
                "random_state"   : HP["random_state"],
                "threshold_pct"  : HP["threshold_pct"],
                "window_size"    : WINDOW_SIZE,
                "n_channels"     : len(MAIN_CHANNELS),
                "n_features"     : metrics.get("n_features", 0),
            })

            # Metrics
            mlflow.log_metrics({
                "precision"       : metrics["precision"],
                "recall"          : metrics["recall"],
                "f1"              : metrics["f1"],
                "auc"             : metrics["auc"],
                "threshold"       : metrics["threshold"],
                "tp"              : metrics["tp"],
                "fp"              : metrics["fp"],
                "tn"              : metrics["tn"],
                "fn"              : metrics["fn"],
                "n_test_windows"  : metrics["n_test_windows"],
                "n_test_abnormal" : metrics["n_test_abnormal"],
            })

            # Log model sklearn
            model_path = MODEL_DIR / "isolation_forest.pkl"
            if model_path.exists():
                iforest_loaded = __import__("joblib").load(model_path)
                mlflow.sklearn.log_model(iforest_loaded, "isolation_forest")

            # Log artifacts (plot hasil evaluasi)
            for fig_name in [
                "iforest_evaluation.png",
                "iforest_anomaly_timeline.png",
                "iforest_score_distribution.png",
                "iforest_feature_importance.png",
            ]:
                fig_path = RESULTS_DIR / "figures" / fig_name
                if fig_path.exists():
                    mlflow.log_artifact(str(fig_path), artifact_path="figures")

            print(f"  [MLflow] Run logged: isolation_forest | "
                  f"F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f}")

    except Exception as e:
        print(f"  [MLflow] Logging dilewati: {e}")


def main():
    parser = argparse.ArgumentParser(description="SHMS Isolation Forest")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    if args.train or args.retrain:
        iforest, scaler, threshold = train(HP, MODEL_DIR, RESULTS_DIR)
        evaluate(iforest, scaler, threshold, MODEL_DIR, RESULTS_DIR)
    elif args.eval:
        iforest, scaler, threshold = load_model(MODEL_DIR)
        evaluate(iforest, scaler, threshold, MODEL_DIR, RESULTS_DIR)
    else:
        run_phase3b()


if __name__ == "__main__":
    main()
