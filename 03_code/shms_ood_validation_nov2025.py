"""
SHMS Bridge Anomaly Detection
shms_ood_validation_nov2025.py — OOD Validation Data November 2025

ARSITEKTUR: DAY-BY-DAY STREAMING (bukan load semua sekaligus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Peak RAM = 1 hari = ~1.4 GB (aman untuk ZBook 16 GB RAM)
490,080 windows TIDAK pernah dimuat sekaligus.

Alur per hari:
  1. Load X_day (.npy) ~1.4 GB
  2. Normalisasi X_day chunked (500 window/chunk) ~1.4 GB
  3. Predict LSTM / IForest / GNN
  4. Simpan skor ke list kecil (~4 MB)
  5. Del X_day → RAM bebas kembali ke baseline

LABELING STRATEGY (Mixed y=0 dan y=1):
  y = 0 (NORMAL)  : 1 Nov - (cutoff-1) Nov
  y = 1 (ANOMALI) : cutoff Nov - 30 Nov
  Default cutoff = 14 (penurunan mendadak -40.5 kN, z = -25.6 sigma)

Cara pakai:
  # Run penuh 30 hari:
  python shms_ood_validation_nov2025.py

  # Quick test 5 hari (HARUS mencakup cutoff agar ada y=0 dan y=1):
  python shms_ood_validation_nov2025.py --days 2025-11-12,2025-11-13,2025-11-14,2025-11-15,2025-11-16

  # Cutoff berbeda:
  python shms_ood_validation_nov2025.py --cutoff 8

Output:
  05_results/nov2025/nov2025_ood_report.csv
  05_results/nov2025/nov2025_score_timeline.png
  05_results/nov2025/nov2025_score_dist.png
  05_results/nov2025/nov2025_roc_curves.png
"""

import argparse
import gc
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from shms_config import (
    ALL_CHANNELS, MAIN_CHANNELS, CONTEXT_CHANNELS,
    MODEL_DIR, RESULTS_DIR,
    get_processed_dir,
)
from shms_phase2_preprocessing import AdaptiveNormalizer


# ─────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────

NOV2025_START  = date(2025, 11, 1)
NOV2025_END    = date(2025, 11, 30)
ANOMALY_CUTOFF = 14

N_MAIN = len(MAIN_CHANNELS)


def get_processed_nov2025() -> Path:
    return get_processed_dir() / "nov2025"


def get_results_nov2025() -> Path:
    d = RESULTS_DIR / "nov2025"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────
# NORMALIZER — load sekali, ringan
# ─────────────────────────────────────────────────────────

def load_normalizer():
    norm_path = get_processed_dir() / "p2_normalizer_stats.csv"
    if not norm_path.exists():
        print(f"  [WARN] Normalizer tidak ditemukan: {norm_path}")
        return None
    norm = AdaptiveNormalizer.load(norm_path)
    print(f"  Normalizer : {norm_path.name} ({len(norm.stats_)} channel)")
    return norm


def normalize_day(X_day: np.ndarray, norm) -> np.ndarray:
    """
    Normalisasi 1 hari (n_win, 1000, 20) secara chunked.
    CHUNK=500 window ~ 38 MB per chunk.

    X_day bisa berupa numpy.memmap (dari mmap_mode='r').
    np.array(chunk) mengkonversi memmap slice ke RAM array biasa.
    Memmap hanya membaca dari disk saat chunk diakses — hemat RAM.
    """
    if norm is None:
        # Return array biasa (bukan memmap) agar aman diproses model
        return np.array(X_day, dtype=np.float32)

    n_win, win_size, n_ch = X_day.shape
    use_names = [c for c in MAIN_CHANNELS + CONTEXT_CHANNELS
                 if c in ALL_CHANNELS][:n_ch]
    CHUNK = 500
    X_out = np.empty((n_win, win_size, n_ch), dtype=np.float32)

    for start in range(0, n_win, CHUNK):
        end   = min(start + CHUNK, n_win)
        # np.array() = copy memmap slice ke RAM — hanya 38 MB per chunk
        chunk  = np.array(X_day[start:end], dtype=np.float32)
        flat   = chunk.reshape(-1, n_ch)
        df_fl  = pd.DataFrame(flat, columns=use_names)
        df_nm  = norm.transform(df_fl)
        normed = df_nm.values.reshape(
            end - start, win_size, n_ch
        ).astype(np.float32)
        np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        X_out[start:end] = normed
        del chunk, flat, df_fl, df_nm, normed   # bebaskan RAM chunk segera

    return X_out


# ─────────────────────────────────────────────────────────
# MODEL — load sekali, tetap di RAM
# ─────────────────────────────────────────────────────────

def load_models() -> dict:
    models = {}

    print("  Loading LSTM Autoencoder...", end="")
    try:
        from shms_phase3a_lstm import load_model as lstm_load
        m, thr = lstm_load(MODEL_DIR)
        models["LSTM"] = {"model": m, "threshold": thr}
        print(f"  threshold={thr:.4f}")
    except Exception as e:
        print(f"  skip ({e})")
        models["LSTM"] = None

    print("  Loading Isolation Forest...", end="")
    try:
        from shms_phase3b_iforest import load_model as if_load
        clf, scaler, thr = if_load(MODEL_DIR)
        models["IForest"] = {"model": clf, "scaler": scaler, "threshold": thr}
        print(f"  threshold={thr:.5f}")
    except Exception as e:
        print(f"  skip ({e})")
        models["IForest"] = None

    print("  Loading GNN Autoencoder...", end="")
    try:
        from shms_phase3c_gnn import load_model as gnn_load
        import pickle

        try:
            m, gd, thr = gnn_load(MODEL_DIR)
        except FileNotFoundError:
            import torch as _t, numpy as _n
            ckpt = _t.load(str(MODEL_DIR / "gnn_model_best.pt"), map_location="cpu")
            ei  = _n.array(ckpt["edge_index"])
            ew  = _n.array(ckpt["edge_weight"])
            adj = _n.array(ckpt["adj_matrix"])
            gd  = (ei, ew, adj)
            with open(str(MODEL_DIR / "gnn_graph_data.pkl"), "wb") as _f:
                pickle.dump({"edge_index": ei, "edge_weight": ew,
                             "adj_matrix": adj}, _f)
            m, _, thr = gnn_load(MODEL_DIR)
        models["GNN"] = {"model": m, "graph_data": gd, "threshold": thr}
        print(f"  threshold={thr:.4f}")
    except Exception as e:
        print(f"  skip ({e})")
        models["GNN"] = None

    n_ok = sum(1 for v in models.values() if v is not None)
    print(f"\n  Model tersedia: {n_ok}/3")
    return models


# ─────────────────────────────────────────────────────────
# PREDICT 1 HARI
# ─────────────────────────────────────────────────────────

def predict_chunk(chunk_norm: np.ndarray, models: dict) -> dict:
    """
    Predict satu chunk (500 windows) dari ketiga model.
    chunk_norm shape: (n_chunk, 1000, 20) — sudah dinormalisasi.
    Return: dict skor per model untuk chunk ini.
    Peak RAM = chunk_norm (~38 MB) + model weights (~1 GB) saja.
    """
    scores = {}

    if models.get("LSTM"):
        try:
            from shms_phase3a_lstm import predict as pred
            res = pred(chunk_norm[:, :, :N_MAIN],
                       model=models["LSTM"]["model"],
                       threshold=models["LSTM"]["threshold"])
            scores["LSTM"] = res["scores"].astype(np.float32)
        except Exception as e:
            scores["LSTM"] = np.full(len(chunk_norm), np.nan, dtype=np.float32)

    if models.get("IForest"):
        try:
            from shms_phase3b_iforest import predict as pred
            res = pred(chunk_norm[:, :, :N_MAIN],
                       iforest=models["IForest"]["model"],
                       scaler=models["IForest"]["scaler"],
                       threshold=models["IForest"]["threshold"])
            scores["IForest"] = res["scores"].astype(np.float32)
        except Exception as e:
            scores["IForest"] = np.full(len(chunk_norm), np.nan, dtype=np.float32)

    if models.get("GNN"):
        try:
            from shms_phase3c_gnn import predict as pred
            res    = pred(chunk_norm,
                          model=models["GNN"]["model"],
                          graph_data=models["GNN"]["graph_data"],
                          threshold=models["GNN"]["threshold"])
            sc_gnn = res["scores"].astype(np.float32)
            n_nan  = int(np.isnan(sc_gnn).sum())
            if n_nan > 0:
                np.nan_to_num(sc_gnn, nan=0.0, posinf=1.0,
                              neginf=0.0, copy=False)
            scores["GNN"] = sc_gnn
        except Exception as e:
            scores["GNN"] = np.full(len(chunk_norm), np.nan, dtype=np.float32)

    return scores


# ─────────────────────────────────────────────────────────
# LOOP STREAMING
# ─────────────────────────────────────────────────────────

def run_streaming(processed_dir, cutoff_day, target_dates, norm, models):
    """
    FULLY STREAMING: normalisasi + predict dilakukan PER CHUNK (500 window).
    File .npy dibuka dengan mmap_mode='r' — tidak pernah di-copy ke RAM penuh.

    Peak RAM per iterasi:
      chunk_norm : 500 × 1000 × 20 × 4 bytes = ~38 MB
      Model LSTM : ~500 MB (weights tetap di RAM sepanjang loop)
      Model IF   : ~300 MB
      Model GNN  : ~300 MB
      OS + lain  : ~2 GB
      ─────────────────────────────────
      TOTAL PEAK : ~3.1 GB  (aman untuk ZBook 16 GB)

    Tidak ada lagi array X_norm (1.3 GB) atau X_day penuh di RAM.
    """
    CHUNK      = 500   # windows per chunk — sesuaikan ke bawah jika masih OOM
    model_names = [n for n, v in models.items() if v is not None]
    use_names   = [c for c in MAIN_CHANNELS + CONTEXT_CHANNELS
                   if c in ALL_CHANNELS]

    accum      = {n: [] for n in model_names}
    y_list     = []
    dates_list = []
    n_normal = n_anomaly = 0

    for day_obj in sorted(target_dates):
        date_str = day_obj.strftime("%Y%m%d")
        x_path   = processed_dir / f"{date_str}_X.npy"
        if not x_path.exists():
            print(f"  [{date_str}] tidak ada, skip")
            continue

        label   = 1 if day_obj.day >= cutoff_day else 0
        lbl_str = "ANOMALI" if label == 1 else "normal "

        # Buka file sebagai memory-map — TIDAK copy ke RAM
        try:
            X_mmap = np.load(x_path, mmap_mode='r')
            if X_mmap.ndim != 3 or X_mmap.shape[0] == 0:
                raise ValueError(f"shape tidak valid: {X_mmap.shape}")
        except Exception as e:
            file_mb = x_path.stat().st_size / 1024**2
            print(f"  [{date_str}] ⚠️  SKIP ({file_mb:.1f} MB): {e}")
            print(f"             Repair: python shms_batch_processor_nov2025.py "
                  f"--date {day_obj} --force")
            continue

        n_win_day = X_mmap.shape[0]
        n_ch      = X_mmap.shape[2]
        ch_names  = use_names[:n_ch]

        print(f"  [{date_str}] y={label} ({lbl_str}) | "
              f"{n_win_day:,} win", end=" | ")

        # Akumulator skor untuk hari ini
        day_scores = {n: [] for n in model_names}

        # ── FULLY STREAMING: chunk-by-chunk ──────────────────────────
        for start in range(0, n_win_day, CHUNK):
            end = min(start + CHUNK, n_win_day)

            # 1. Copy chunk kecil dari mmap ke RAM (~38 MB)
            chunk = np.array(X_mmap[start:end], dtype=np.float32)

            # 2. Normalisasi chunk
            if norm is not None:
                flat  = chunk.reshape(-1, n_ch)
                df_fl = pd.DataFrame(flat, columns=ch_names)
                df_nm = norm.transform(df_fl)
                chunk = df_nm.values.reshape(
                    end - start, X_mmap.shape[1], n_ch
                ).astype(np.float32)
                np.nan_to_num(chunk, nan=0.0, posinf=0.0,
                              neginf=0.0, copy=False)
                del flat, df_fl, df_nm

            # 3. Predict semua model pada chunk ini
            sc_chunk = predict_chunk(chunk, models)
            del chunk
            gc.collect()

            # 4. Akumulasi skor chunk
            for n in model_names:
                sc = sc_chunk.get(n)
                if sc is not None:
                    day_scores[n].append(sc)
                else:
                    day_scores[n].append(
                        np.full(end - start, np.nan, dtype=np.float32))

        # Tutup mmap dan bebaskan reference
        del X_mmap
        gc.collect()
        # ─────────────────────────────────────────────────────────────

        # Gabung skor semua chunk untuk hari ini
        for n in model_names:
            if day_scores[n]:
                accum[n].append(np.concatenate(day_scores[n]))
            else:
                accum[n].append(np.full(n_win_day, np.nan, dtype=np.float32))

        y_list.extend([label] * n_win_day)
        dates_list.extend([day_obj] * n_win_day)

        if label == 0:
            n_normal  += n_win_day
        else:
            n_anomaly += n_win_day

        print("done")

    # Gabung semua hari
    all_scores = {n: np.concatenate(v) if v else np.array([])
                  for n, v in accum.items()}
    y_true = np.array(y_list, dtype=np.int8)

    print(f"\n  Total      : {len(y_true):,} windows")
    print(f"  Normal y=0 : {n_normal:,} ({100*n_normal/max(len(y_true),1):.1f}%)")
    print(f"  Anomali y=1: {n_anomaly:,} ({100*n_anomaly/max(len(y_true),1):.1f}%)")

    return all_scores, y_true, dates_list


# ─────────────────────────────────────────────────────────
# METRIK
# ─────────────────────────────────────────────────────────

def compute_metrics(all_scores, y_true, models):
    rows = []
    for name, sc in all_scores.items():
        if sc is None or len(sc) == 0:
            rows.append({"Model": name, "Status": "N/A"})
            continue

        valid_mask = ~np.isnan(sc)
        sc_v = sc[valid_mask]
        y_v  = y_true[valid_mask]

        if len(sc_v) == 0 or (y_v == 0).sum() == 0 or (y_v == 1).sum() == 0:
            rows.append({"Model": name, "Status": "No mixed labels"})
            continue

        # ── PERBAIKAN BUG 1 ─────────────────────────────────────────────
        # Threshold dari P95 window NORMAL saja (y=0),
        # konsisten dengan training: P95 dari validation set 2026 (semua normal).
        # Sebelumnya: np.percentile(sc_v, 95) → dari semua data (40% normal + 60%
        # anomali) → threshold terlalu tinggi → Recall sangat rendah.
        sc_normal = sc_v[y_v == 0]
        thr_norm  = float(np.percentile(sc_normal, 95))
        y_pred    = (sc_v >= thr_norm).astype(int)
        # ────────────────────────────────────────────────────────────────

        prec   = precision_score(y_v, y_pred, zero_division=0)
        recall = recall_score(y_v, y_pred, zero_division=0)
        f1     = f1_score(y_v, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_v, sc_v)
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_v, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # ── PERBAIKAN BUG 2 ─────────────────────────────────────────────
        # Tambahkan F1-optimal (Youden Index) dari ROC curve.
        # Menunjukkan performa MAKSIMUM model jika threshold di-tune secara optimal.
        # Berguna untuk reviewer yang ingin tahu potensi tertinggi model.
        try:
            from sklearn.metrics import roc_curve as _roc
            fpr_arr, tpr_arr, thr_arr = _roc(y_v, sc_v)
            # Youden index = TPR - FPR, pilih threshold yang maksimalkan ini
            youden_idx  = np.argmax(tpr_arr - fpr_arr)
            thr_optimal = float(thr_arr[youden_idx])
            y_pred_opt  = (sc_v >= thr_optimal).astype(int)
            f1_optimal  = round(f1_score(y_v, y_pred_opt, zero_division=0), 4)
            prec_opt    = round(precision_score(y_v, y_pred_opt, zero_division=0), 4)
            rec_opt     = round(recall_score(y_v, y_pred_opt, zero_division=0), 4)
        except Exception:
            f1_optimal = prec_opt = rec_opt = thr_optimal = float("nan")
        # ────────────────────────────────────────────────────────────────

        rows.append({
            "Model"          : name,
            "Status"         : "OK",
            # Threshold P95 dari normal (konsisten dengan training)
            "Threshold_P95"  : round(thr_norm, 5),
            "Precision"      : round(prec, 4),
            "Recall"         : round(recall, 4),
            "F1"             : round(f1, 4),
            "AUC_ROC"        : round(auc, 4) if not np.isnan(auc) else "N/A",
            "TP"             : int(tp),
            "FP"             : int(fp),
            "FN"             : int(fn),
            "TN"             : int(tn),
            # Metrik optimal (Youden Index) — performa maksimum model
            "Threshold_Opt"  : round(thr_optimal, 5) if not np.isnan(thr_optimal) else "N/A",
            "Precision_Opt"  : prec_opt,
            "Recall_Opt"     : rec_opt,
            "F1_Opt"         : f1_optimal,
            "N_Normal"       : int((y_v == 0).sum()),
            "N_Anomaly"      : int((y_v == 1).sum()),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────

def make_plots(all_scores, y_true, dates_list, cutoff_day,
               metrics_df, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    active = [(n, sc) for n, sc in all_scores.items()
              if sc is not None and len(sc) > 0]
    if not active:
        return

    colors = {"LSTM": "#2166AC", "IForest": "#1B7837", "GNN": "#762A83"}
    cutoff_idx = next((i for i, d in enumerate(dates_list)
                       if d.day >= cutoff_day), None)

    # Sumbu x: tick per hari
    day_ticks, day_labels = [], []
    for i, d in enumerate(dates_list):
        if i == 0 or dates_list[i].day != dates_list[i-1].day:
            day_ticks.append(i)
            day_labels.append(f"{d.day}\nNov")

    n_models = len(active)

    # ── Plot 1: Timeline ─────────────────────────────────
    fig, axes = plt.subplots(n_models, 1,
                             figsize=(14, 3.8 * n_models),
                             sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, sc) in zip(axes, active):
        sc_plot = np.where(np.isnan(sc), np.nan, sc)
        col     = colors.get(name, "#888")

        ax.scatter(np.where(y_true == 0)[0],
                   sc_plot[y_true == 0],
                   c="#2166AC", s=1.5, alpha=0.3,
                   label="Normal (y=0)", rasterized=True)
        ax.scatter(np.where(y_true == 1)[0],
                   sc_plot[y_true == 1],
                   c="#E24B4A", s=1.5, alpha=0.3,
                   label="Anomali (y=1)", rasterized=True)

        row = metrics_df[metrics_df["Model"] == name]
        if len(row) and row.iloc[0].get("Status") == "OK":
            thr_v = float(row.iloc[0]["Threshold_P95"])
            ax.axhline(thr_v, color=col, lw=1.2, ls=":",
                       alpha=0.9, label=f"Threshold P95={thr_v:.3f}")
            r = row.iloc[0]
            title = (f"{name}  —  "
                     f"Precision={r['Precision']:.3f}  "
                     f"Recall={r['Recall']:.3f}  "
                     f"F1={r['F1']:.3f}  "
                     f"AUC={r['AUC_ROC']}")
        else:
            title = name

        if cutoff_idx is not None:
            ax.axvline(cutoff_idx, color="#D62728", lw=2, ls="--",
                       label=f"{cutoff_day} Nov (cutoff)")
            ax.axvspan(cutoff_idx, len(sc), alpha=0.04, color="#D62728")

        ax.set_ylabel("Anomaly Score", fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left", markerscale=3)
        ax.grid(True, alpha=0.15)
        ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xticks(day_ticks[::2])
    axes[-1].set_xticklabels(day_labels[::2], fontsize=8)
    axes[-1].set_xlabel("Tanggal (November 2025)", fontsize=9)
    fig.suptitle(
        "Anomaly Score Timeline — November 2025  |  "
        "Biru = normal  |  Merah = anomali  |  Garis putus = cutoff",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    p1 = save_dir / "nov2025_score_timeline.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot 1: {p1.name}")

    # ── Plot 2: Distribusi ────────────────────────────────
    fig, axes = plt.subplots(1, n_models,
                             figsize=(5 * n_models, 4),
                             squeeze=False)
    for ax, (name, sc) in zip(axes[0], active):
        sc_n = sc[(y_true == 0) & ~np.isnan(sc)]
        sc_a = sc[(y_true == 1) & ~np.isnan(sc)]
        ax.hist(sc_n, bins=50, alpha=0.65, color="#2166AC",
                density=True, label=f"Normal ({len(sc_n):,})")
        ax.hist(sc_a, bins=50, alpha=0.65, color="#E24B4A",
                density=True, label=f"Anomali ({len(sc_a):,})")
        row = metrics_df[metrics_df["Model"] == name]
        if len(row) and row.iloc[0].get("Status") == "OK":
            thr_v = float(row.iloc[0]["Threshold_P95"])
            ax.axvline(thr_v, color="black", lw=1.5, ls="--",
                       label=f"Thr={thr_v:.3f}")
        ax.set_xlabel("Anomaly Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Score Distribution: Normal vs Anomali — November 2025",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    p2 = save_dir / "nov2025_score_dist.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot 2: {p2.name}")

    # ── Plot 3: ROC Curves ────────────────────────────────
    if (y_true == 0).any() and (y_true == 1).any():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5,
                label="Random (AUC=0.50)")
        for name, sc in active:
            sc_v = np.where(np.isnan(sc), 0.5, sc)
            try:
                fpr_a, tpr_a, _ = roc_curve(y_true, sc_v)
                auc_v = roc_auc_score(y_true, sc_v)
                col   = colors.get(name, "#888")
                ax.plot(fpr_a, tpr_a, lw=2.2, color=col,
                        label=f"{name}  (AUC={auc_v:.4f})")
            except Exception:
                pass
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(
            "ROC Curves — November 2025\n"
            "(AUC valid: ada y=0 dan y=1)",
            fontsize=10, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        p3 = save_dir / "nov2025_roc_curves.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot 3: {p3.name}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OOD Validation Nov 2025 — day-by-day streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # Run penuh 30 hari:
  python shms_ood_validation_nov2025.py

  # Quick test 5 hari (ada y=0 DAN y=1):
  python shms_ood_validation_nov2025.py --days 2025-11-12,2025-11-13,2025-11-14,2025-11-15,2025-11-16

  # Cutoff 8 Nov (awal gradual drift):
  python shms_ood_validation_nov2025.py --cutoff 8

  # Tanpa normalisasi:
  python shms_ood_validation_nov2025.py --no-normalize
        """
    )
    parser.add_argument("--cutoff",       type=int, default=ANOMALY_CUTOFF,
                        help=f"Hari anomali mulai (default: {ANOMALY_CUTOFF})")
    parser.add_argument("--days",         type=str, default=None,
                        help="Daftar tanggal: 2025-11-12,2025-11-14,...")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip normalisasi")
    args = parser.parse_args()

    t0          = time.time()
    processed   = get_processed_nov2025()
    results_dir = get_results_nov2025()

    print("\n" + "="*65)
    print("  OOD VALIDATION — NOVEMBER 2025")
    print("  Arsitektur: Day-by-Day Streaming (peak RAM ~1.4 GB/hari)")
    print("="*65)
    print(f"\n  Processed dir : {processed}")
    print(f"  Model dir     : {MODEL_DIR}")
    print(f"  Results dir   : {results_dir}")
    print(f"  Cutoff        : {args.cutoff} November 2025")
    print(f"  Labeling      : y=0 (1-{args.cutoff-1} Nov) | "
          f"y=1 ({args.cutoff}-30 Nov)")

    # Tentukan hari yang diproses
    if args.days:
        target_dates = []
        for ds in args.days.split(","):
            ds = ds.strip()
            y2, m, d = int(ds[:4]), int(ds[5:7]), int(ds[8:10])
            target_dates.append(date(y2, m, d))
        print(f"\n  Mode : hari spesifik ({len(target_dates)} hari)")
        for td in sorted(target_dates):
            lbl = "ANOMALI" if td.day >= args.cutoff else "normal "
            print(f"    {td}  y={'1' if td.day >= args.cutoff else '0'} ({lbl})")
    else:
        target_dates = []
        cur = NOV2025_START
        while cur <= NOV2025_END:
            if (processed / f"{cur.strftime('%Y%m%d')}_X.npy").exists():
                target_dates.append(cur)
            cur += timedelta(days=1)
        print(f"\n  Mode : semua hari ({len(target_dates)} hari tersedia)")

    # Step 1: Load normalizer
    print(f"\n[1/4] Load normalizer...")
    norm = None if args.no_normalize else load_normalizer()

    # Step 2: Load model
    print(f"\n[2/4] Load model...")
    models = load_models()
    if all(v is None for v in models.values()):
        print("\n  Tidak ada model tersedia. Hentikan.")
        sys.exit(1)

    # Step 3: Streaming per hari
    print(f"\n[3/4] Streaming per hari...")
    all_scores, y_true, dates_list = run_streaming(
        processed, args.cutoff, target_dates, norm, models
    )

    # Step 4: Metrik + Output
    print(f"\n[4/4] Hitung metrik dan buat plot...")
    metrics_df = compute_metrics(all_scores, y_true, models)

    # Tabel hasil
    print(f"\n  {'='*78}")
    print(f"  HASIL OOD VALIDATION — NOVEMBER 2025")
    print(f"  {'='*78}")
    print(f"  Normal  (y=0): {(y_true==0).sum():,} windows "
          f"(1-{args.cutoff-1} Nov)")
    print(f"  Anomali (y=1): {(y_true==1).sum():,} windows "
          f"({args.cutoff}-30 Nov)")

    # ── Tabel A: Threshold P95 dari normal (konsisten dengan training) ──
    print(f"\n  [A] Threshold P95 dari data NORMAL (1-{args.cutoff-1} Nov) "
          f"— konsisten dengan training:")
    print(f"  {'Model':<10} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'AUC':>7} {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>8}")
    print(f"  {'-'*72}")
    for _, r in metrics_df.iterrows():
        if r.get("Status") != "OK":
            print(f"  {r['Model']:<10}  ({r.get('Status', 'N/A')})")
            continue
        print(f"  {r['Model']:<10} "
              f"{r['Precision']:>7.4f} {r['Recall']:>7.4f} "
              f"{r['F1']:>7.4f} {str(r['AUC_ROC']):>7} "
              f"{r['TP']:>8,} {r['FP']:>8,} "
              f"{r['FN']:>8,} {r['TN']:>8,}")

    # ── Tabel B: Threshold optimal (Youden Index) — potensi maksimum ──
    print(f"\n  [B] Threshold OPTIMAL (Youden Index) "
          f"— performa maksimum model:")
    print(f"  {'Model':<10} {'Prec_Opt':>9} {'Rec_Opt':>8} "
          f"{'F1_Opt':>8} {'Thr_Opt':>9}")
    print(f"  {'-'*48}")
    for _, r in metrics_df.iterrows():
        if r.get("Status") != "OK":
            continue
        print(f"  {r['Model']:<10} "
              f"{str(r['Precision_Opt']):>9} "
              f"{str(r['Recall_Opt']):>8} "
              f"{str(r['F1_Opt']):>8} "
              f"{str(r['Threshold_Opt']):>9}")

    csv_path = results_dir / "nov2025_ood_report.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Report : {csv_path}")

    make_plots(all_scores, y_true, dates_list,
               args.cutoff, metrics_df, results_dir)

    # Interpretasi berdasarkan AUC (metrik yang tidak dipengaruhi threshold)
    print(f"\n  INTERPRETASI (berdasarkan AUC — tidak terpengaruh threshold):")
    for _, r in metrics_df.iterrows():
        if r.get("Status") != "OK":
            continue
        try:
            auc_v = float(r["AUC_ROC"])
        except (ValueError, TypeError):
            auc_v = 0.0
        f1_opt = r.get("F1_Opt", "N/A")
        if auc_v >= 0.85:
            interp = "SANGAT BAIK — diskriminasi kuat antara normal dan anomali"
        elif auc_v >= 0.70:
            interp = "BAIK — model mampu membedakan subtle in-service anomaly"
        elif auc_v >= 0.55:
            interp = "CUKUP — diskriminasi lemah, perlu perbaikan fitur"
        else:
            interp = "TIDAK SENSITIF — model tidak mendeteksi jenis anomali ini"
        print(f"  {r['Model']:<10}: AUC={r['AUC_ROC']}, "
              f"F1_opt={f1_opt} → {interp}")

    print(f"\n  Total waktu : {(time.time()-t0)/60:.1f} menit")
    print("="*65)


if __name__ == "__main__":
    main()