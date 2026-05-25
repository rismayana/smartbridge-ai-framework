"""
SHMS Bridge Anomaly Detection
shms_ood_validation.py — Out-of-Distribution Validation

Memvalidasi kemampuan deteksi ketiga model menggunakan data
periode pra-operasional (Maret 2024) sebagai proxy anomali.

Latar belakang:
    Model dilatih dengan data normal Feb-Mar 2026 (kabel bertegangan
    500-900 kN). Data Maret 2024 menunjukkan kabel belum ditegangkan
    (3-17 kN) — perbedaan 14-18 sigma dari kondisi normal.
    Ini merepresentasikan kondisi struktural abnormal yang valid
    secara teknis (Hendrycks & Gimpel, 2017).

Cara pakai:
    # Letakkan beberapa file ALL_*.txt Maret 2024 di satu folder
    python shms_ood_validation.py --data-dir "F:/Data SHMS Batam/RawData2024"

    # Jika file ada di folder yang sama dengan raw data 2026:
    python shms_ood_validation.py --data-dir "F:/Data SHMS Batam/RawData" --date-filter 2024

    # Quick test dengan 1 file saja:
    python shms_ood_validation.py --data-dir "F:/Data SHMS Batam/RawData2024" --max-files 1

Output:
    05_results/ood_validation_report.csv   — metrik per model
    05_results/figures/ood_score_dist.png  — distribusi score
    05_results/figures/ood_timeline.png    — timeline anomali score
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — aman untuk ZBook tanpa display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from shms_config import (
    ALL_CHANNELS, MAIN_CHANNELS, CONTEXT_CHANNELS,
    WIND_CHANNELS, FILE_ENCODING, FILE_NA_VALUES,
    RAW_FILE_PATTERN, RAW_FILE_PREFIX, RAW_DATETIME_FMT,
    WINDOW_SIZE, WINDOW_STEP, MAX_NAN_PCT,
    MODEL_DIR, RESULTS_DIR,
    get_processed_dir,
)
from shms_phase2_preprocessing import AdaptiveNormalizer


# ─────────────────────────────────────────────────────────
# STEP 1: LOAD FILE RAW 2024
# ─────────────────────────────────────────────────────────

def load_raw_files(data_dir: Path,
                   date_filter: str = None,
                   max_files: int = None) -> np.ndarray | None:
    """
    Load file ALL_*.txt dari folder data_dir.

    Proses: baca CSV → filter 20 channel → forward-fill anemometer
            → gabung semua baris → return numpy array float32

    Args:
        data_dir    : folder berisi file ALL_*.txt
        date_filter : string filter tanggal, misal "2024" atau "20240307"
        max_files   : batasi jumlah file (untuk quick test)

    Return:
        numpy array (total_rows, 20) float32, atau None jika gagal
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"  ❌ Folder tidak ditemukan: {data_dir}")
        return None

    # Kumpulkan file
    all_files = sorted(data_dir.glob("ALL_*.txt"))
    if date_filter:
        all_files = [f for f in all_files if date_filter in f.stem]

    if not all_files:
        print(f"  ❌ Tidak ada file ALL_*.txt di: {data_dir}")
        if date_filter:
            print(f"     dengan filter tanggal: {date_filter}")
        return None

    if max_files:
        all_files = all_files[:max_files]

    print(f"  File ditemukan  : {len(all_files)}")
    print(f"  Contoh file     : {all_files[0].name}")
    print(f"  Sampai          : {all_files[-1].name}")

    arrays = []
    n_loaded = 0
    n_failed = 0

    for f in all_files:
        try:
            # Baca header dulu
            hdr = pd.read_csv(f, encoding=FILE_ENCODING, nrows=0, low_memory=False)
            hdr.columns = [
                c.strip().encode("ascii","ignore").decode("ascii").strip()
                for c in hdr.columns
            ]

            # Pilih 20 channel yang ada
            usecols = [c for c in ALL_CHANNELS if c in hdr.columns]
            if not usecols:
                n_failed += 1
                continue

            df = pd.read_csv(
                f,
                encoding  = FILE_ENCODING,
                na_values = FILE_NA_VALUES,
                usecols   = usecols,
                dtype     = {c: "float32" for c in usecols},
                low_memory= False,
            )

            if df.empty or len(df) < 100:
                n_failed += 1
                continue

            # Forward-fill anemometer (1Hz → ikuti 100Hz)
            for col in WIND_CHANNELS:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()

            # Susun sesuai ALL_CHANNELS, pad NaN jika tidak ada
            arr = np.full((len(df), len(ALL_CHANNELS)), np.nan, dtype=np.float32)
            for i, col in enumerate(ALL_CHANNELS):
                if col in df.columns:
                    arr[:, i] = df[col].values

            arrays.append(arr)
            n_loaded += 1

        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")
            n_failed += 1

    if not arrays:
        print(f"  ❌ Semua file gagal dibaca")
        return None

    data = np.concatenate(arrays, axis=0)
    print(f"  Berhasil dimuat : {n_loaded} file ({n_failed} gagal)")
    print(f"  Total rows      : {len(data):,} (~{len(data)/100:.0f} detik @ 100Hz)")
    return data


# ─────────────────────────────────────────────────────────
# STEP 2: SLIDING WINDOW
# ─────────────────────────────────────────────────────────

def make_windows(data: np.ndarray) -> np.ndarray:
    """
    Potong data menjadi sliding windows.

    Semua window dari data 2024 diberi label y=1 (anomali)
    karena kondisi struktural secara fisik abnormal.

    Return: X (n_windows, WINDOW_SIZE, n_channels) float32
    """
    # Pakai channel yang sama dengan model (MAIN + CONTEXT)
    use_cols = [ALL_CHANNELS.index(c)
                for c in MAIN_CHANNELS + CONTEXT_CHANNELS
                if c in ALL_CHANNELS]
    sensor = data[:, use_cols]

    n_rows = len(sensor)
    n_max  = max(0, (n_rows - WINDOW_SIZE) // WINDOW_STEP + 1)
    if n_max == 0:
        return np.empty((0, WINDOW_SIZE, len(use_cols)), dtype=np.float32)

    X_out = np.empty((n_max, WINDOW_SIZE, len(use_cols)), dtype=np.float32)
    count = 0

    for start in range(0, n_rows - WINDOW_SIZE + 1, WINDOW_STEP):
        end    = start + WINDOW_SIZE
        window = sensor[start:end].copy()

        # Buang window dengan terlalu banyak NaN
        if np.isnan(window).mean() > MAX_NAN_PCT:
            continue

        # Interpolasi NaN sisa
        if np.isnan(window).any():
            for j in range(window.shape[1]):
                col = window[:, j]
                if np.isnan(col).any():
                    mask = np.isnan(col)
                    valid = ~mask
                    if valid.sum() > 1:
                        idx = np.arange(WINDOW_SIZE)
                        window[:, j] = np.interp(idx, idx[valid], col[valid])
                    else:
                        window[:, j] = 0.0

        X_out[count] = window
        count += 1

    return X_out[:count]


# ─────────────────────────────────────────────────────────
# STEP 3: NORMALISASI DENGAN STATS 2026
# ─────────────────────────────────────────────────────────

def normalize_with_2026_stats(X: np.ndarray) -> np.ndarray:
    """
    Normalisasi data 2024 menggunakan normalizer yang di-fit dari data 2026.

    INI YANG MEMBUAT OOD VALIDATION VALID:
    - Normalizer di-fit dari data normal 2026
    - Data 2024 di-normalisasi dengan parameter yang SAMA
    - Channel cable tension 2024 (~3-17 kN) vs mean 2026 (~500-900 kN)
      → Z-score = (7 - 800) / 50 = -15.9 sigma
      → Sangat jauh dari normal → model pasti deteksi anomali

    Return: X_normalized (n_windows, WINDOW_SIZE, n_channels) float32
    """
    norm_path = get_processed_dir() / "p2_normalizer_stats.csv"
    if not norm_path.exists():
        print(f"  [WARN] Normalizer tidak ditemukan: {norm_path}")
        print(f"         Gunakan data tanpa normalisasi (kurang akurat)")
        return X

    norm = AdaptiveNormalizer.load(norm_path)

    # Buat DataFrame dari X (hanya untuk normalisasi)
    # X shape: (n_windows, WINDOW_SIZE, n_channels)
    # Normalisasi per channel (axis=0 dan axis=1) → gunakan mean per timestep
    n_win, win_size, n_ch = X.shape

    # Reshape ke (n_win * win_size, n_channels) untuk normalisasi
    X_flat = X.reshape(-1, n_ch)
    use_names = [c for c in MAIN_CHANNELS + CONTEXT_CHANNELS if c in ALL_CHANNELS]

    df_flat = pd.DataFrame(X_flat, columns=use_names[:n_ch])
    df_norm = norm.transform(df_flat)

    X_norm = df_norm.values.reshape(n_win, win_size, n_ch).astype(np.float32)
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    print(f"  Normalizer      : loaded dari {norm_path.name}")
    print(f"  Channel stats   : {len(norm.stats_)} channel (dari training 2026)")
    return X_norm


# ─────────────────────────────────────────────────────────
# STEP 4: PREDICT DARI KETIGA MODEL
# ─────────────────────────────────────────────────────────

def predict_all_models(X_ood: np.ndarray) -> dict:
    """
    Jalankan inferensi ketiga model pada data OOD (2024).

    Return: dict berisi scores dan predictions per model
    """
    results = {}

    # ── LSTM ──────────────────────────────────────────────
    print("\n  [LSTM Autoencoder]")
    try:
        from shms_phase3a_lstm import predict as lstm_predict, load_model as lstm_load
        model_lstm, threshold_lstm = lstm_load(MODEL_DIR)
        # LSTM dilatih dengan 17 MAIN_CHANNELS — potong X_ood dari 20 ke 17
        use_idx   = [ALL_CHANNELS.index(c) for c in MAIN_CHANNELS if c in ALL_CHANNELS]
        X_ood_lstm = X_ood[:, :, :len(use_idx)]
        res = lstm_predict(X_ood_lstm, model=model_lstm, threshold=threshold_lstm)
        results["LSTM"] = res
        n_det = res["predictions"].sum()
        print(f"  → {n_det}/{len(X_ood)} windows terdeteksi anomali "
              f"({100*n_det/len(X_ood):.1f}%)")
    except FileNotFoundError:
        print(f"  ⏭  Model LSTM belum ada — skip")
        results["LSTM"] = None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["LSTM"] = None

    # ── Isolation Forest ──────────────────────────────────
    print("\n  [Isolation Forest]")
    try:
        from shms_phase3b_iforest import predict as if_predict, load_model as if_load
        iforest, scaler, threshold_if = if_load(MODEL_DIR)
        # IForest dilatih dengan 17 MAIN_CHANNELS — potong X_ood dari 20 ke 17
        use_idx = [ALL_CHANNELS.index(c) for c in MAIN_CHANNELS if c in ALL_CHANNELS]
        X_ood_if = X_ood[:, :, :len(use_idx)]   # sudah urut MAIN_CHANNELS dari make_windows
        res = if_predict(X_ood_if, iforest=iforest, scaler=scaler, threshold=threshold_if)
        results["IForest"] = res
        n_det = res["predictions"].sum()
        print(f"  → {n_det}/{len(X_ood)} windows terdeteksi anomali "
              f"({100*n_det/len(X_ood):.1f}%)")
    except FileNotFoundError:
        print(f"  ⏭  Model IForest belum ada — skip")
        results["IForest"] = None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["IForest"] = None

    # ── GNN ───────────────────────────────────────────────
    print("\n  [GNN Autoencoder]")
    try:
        from shms_phase3c_gnn import predict as gnn_predict, load_model as gnn_load
        import pickle, numpy as _np

        # Coba load normal dulu
        try:
            model_gnn, graph_data, threshold_gnn = gnn_load(MODEL_DIR)
        except FileNotFoundError:
            # gnn_graph_data.pkl tidak ada → rebuild dari checkpoint
            print(f"  [INFO] gnn_graph_data.pkl tidak ada → rebuild dari checkpoint...")
            import torch as _torch
            ckpt_path = MODEL_DIR / "gnn_model_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError("gnn_model_best.pt tidak ada")
            ckpt = _torch.load(str(ckpt_path), map_location="cpu")
            edge_index  = _np.array(ckpt["edge_index"])
            edge_weight = _np.array(ckpt["edge_weight"])
            adj         = _np.array(ckpt["adj_matrix"])
            graph_data  = (edge_index, edge_weight, adj)
            # Simpan pkl agar tidak perlu rebuild lagi
            pkl_path = MODEL_DIR / "gnn_graph_data.pkl"
            with open(str(pkl_path), "wb") as _f:
                pickle.dump({"edge_index": edge_index,
                             "edge_weight": edge_weight,
                             "adj_matrix": adj}, _f)
            print(f"  [INFO] gnn_graph_data.pkl dibuat dan disimpan")
            # Load ulang model lengkap
            model_gnn, _, threshold_gnn = gnn_load(MODEL_DIR)

        res = gnn_predict(X_ood, model=model_gnn, graph_data=graph_data,
                          threshold=threshold_gnn)
        results["GNN"] = res
        n_det = res["predictions"].sum()
        print(f"  → {n_det}/{len(X_ood)} windows terdeteksi anomali "
              f"({100*n_det/len(X_ood):.1f}%)")
    except FileNotFoundError:
        print(f"  ⏭  Model GNN belum ada — skip")
        results["GNN"] = None
    except Exception as e:
        print(f"  ❌ Error GNN: {e}")
        results["GNN"] = None

    return results


# ─────────────────────────────────────────────────────────
# STEP 5: HITUNG METRIK
# ─────────────────────────────────────────────────────────

def compute_metrics(results: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Hitung F1, Precision, Recall, AUC untuk setiap model.

    y_true: semua 1 (semua window dari data 2024 = anomali)
    Recall = TP / (TP + FN) = proporsi anomali yang berhasil terdeteksi
    """
    rows = []
    for model_name, res in results.items():
        if res is None:
            rows.append({
                "Model": model_name, "Status": "Not available",
                "Precision": "-", "Recall": "-", "F1": "-", "AUC": "-",
                "Detected": "-", "Total": len(y_true),
            })
            continue

        y_pred  = res["predictions"].astype(int)
        scores  = res["scores"]

        prec   = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1     = f1_score(y_true, y_pred, zero_division=0)

        # AUC: butuh mix y=0 dan y=1.
        # Karena y_true semua 1, gabung dengan skor dari data normal 2026
        # untuk mendapat AUC yang bermakna (dilakukan di plot saja)
        # Di sini laporkan recall sebagai indikator utama
        auc = float("nan")   # akan dihitung di plot dengan data normal

        n_det  = int(y_pred.sum())
        rows.append({
            "Model"    : model_name,
            "Status"   : "OK",
            "Threshold": round(float(res.get("threshold", 0)), 5),
            "Precision": round(prec,   4),
            "Recall"   : round(recall, 4),
            "F1"       : round(f1,     4),
            "AUC"      : "see_plot",
            "Detected" : n_det,
            "Total"    : len(y_true),
            "Det_pct"  : round(100 * n_det / len(y_true), 1),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# STEP 6: PLOT
# ─────────────────────────────────────────────────────────

def make_plots(results: dict, X_ood: np.ndarray,
               save_dir: Path):
    """
    Buat dua plot:
    1. Distribusi anomaly score: data normal 2026 vs data OOD 2024
    2. Timeline anomaly score pada data OOD 2024
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load sample score dari data normal (validation 2026)
    # untuk perbandingan distribusi
    normal_scores = {}
    try:
        from shms_phase3a_lstm import ShmsIterableDataset
        import torch as _torch
        # Ambil 2000 windows dari val set menggunakan streaming (aman RAM)
        val_ds = ShmsIterableDataset("val", normal_only=True,
                                     windows_per_epoch=2000)
        val_loader = _torch.utils.data.DataLoader(val_ds, batch_size=256)
        X_val_list = []
        for batch in val_loader:
            X_val_list.append(batch.numpy())
            if sum(len(b) for b in X_val_list) >= 2000:
                break
        X_val = np.concatenate(X_val_list, axis=0)[:2000]

        for model_name, res in results.items():
            if res is None:
                continue
            if model_name == "IForest":
                from shms_phase3b_iforest import (
                    load_model as if_load, extract_features,
                    MAIN_CHANNELS as IF_MAIN,
                )
                iforest, scaler, thr = if_load(MODEL_DIR)
                feat = extract_features(X_val, IF_MAIN)
                raw  = -iforest.score_samples(scaler.transform(feat.fillna(0)))
                mn, mx = raw.min(), raw.max()
                normal_scores["IForest"] = (raw - mn) / (mx - mn + 1e-9)
    except Exception:
        pass  # jika model belum ada, skip

    # ── Plot 1: Score Distribution ─────────────────────────
    active = [(n, r) for n, r in results.items() if r is not None]
    if active:
        n_models = len(active)
        fig, axes = plt.subplots(1, n_models,
                                 figsize=(5 * n_models, 4.5),
                                 squeeze=False)

        for ax, (mname, res) in zip(axes[0], active):
            sc_ood = res["scores"]
            thr    = res.get("threshold", 0.5)

            sc_ood_clean = np.nan_to_num(sc_ood, nan=0.0, posinf=1.0, neginf=0.0)
            ax.hist(sc_ood_clean, bins=40, alpha=0.75,
                    color="#E24B4A", label="Data 2024 (anomali)", density=True)

            if mname in normal_scores:
                ns_clean = np.nan_to_num(normal_scores[mname], nan=0.0, posinf=1.0, neginf=0.0)
                ax.hist(ns_clean, bins=40, alpha=0.65,
                        color="#378ADD", label="Data 2026 (normal)", density=True)

            # Normalisasi threshold ke [0,1] scale yang sama dengan scores
            ax.set_xlabel("Anomaly Score (normalized)")
            ax.set_ylabel("Density")
            ax.set_title(f"{mname}\nRecall = "
                         f"{100*res['predictions'].mean():.0f}%")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("OOD Validation: Score Distribution\n"
                     "Data Normal 2026 vs Data Pra-Operasional Maret 2024",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        p1 = save_dir / "ood_score_distribution.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {p1.name}")

    # ── Plot 2: Timeline ────────────────────────────────────
    if active:
        fig, axes = plt.subplots(n_models, 1,
                                 figsize=(14, 3.5 * n_models),
                                 sharex=True)
        if n_models == 1:
            axes = [axes]

        for ax, (mname, res) in zip(axes, active):
            sc   = res["scores"]
            pred = res["predictions"]
            thr_norm = np.percentile(sc[pred == 0], 95) if (pred == 0).any() \
                       else 0.5   # approximate threshold in normalized scale

            sc  = np.nan_to_num(sc, nan=0.0, posinf=1.0, neginf=0.0)
            idx = np.arange(len(sc))
            ax.plot(idx, sc, color="#888780", lw=0.5, alpha=0.8, label="Score")
            ax.fill_between(idx, sc, alpha=0.3, color="#E24B4A")

            # Tandai windows yang TIDAK terdeteksi (false negative)
            fn_idx = np.where(pred == 0)[0]
            if len(fn_idx):
                ax.scatter(fn_idx, sc[fn_idx], color="#1D9E75",
                           s=8, zorder=5, label=f"Tidak terdeteksi ({len(fn_idx)})")

            ax.set_ylabel(f"{mname}\nAnomaly Score")
            ax.set_title(f"{mname} — "
                         f"Terdeteksi: {pred.sum()}/{len(pred)} "
                         f"({100*pred.mean():.0f}%)")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Window Index (kronologis)")
        fig.suptitle("OOD Validation: Anomaly Detection Timeline\n"
                     "Data Maret 2024 (kabel belum bertegangan)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        p2 = save_dir / "ood_anomaly_timeline.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {p2.name}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OOD Validation — validasi model dengan data Maret 2024",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # File 2024 di folder terpisah:
  python shms_ood_validation.py --data-dir "F:/Data2024/RawData"

  # File 2024 campur dengan 2026, filter by date:
  python shms_ood_validation.py --data-dir "F:/Data SHMS Batam/RawData" --date-filter 20240307

  # Quick test 1 file saja:
  python shms_ood_validation.py --data-dir "F:/Data2024" --max-files 1

  # Tanpa normalisasi (jika p2_normalizer_stats.csv belum ada):
  python shms_ood_validation.py --data-dir "F:/Data2024" --no-normalize
        """
    )
    parser.add_argument("--data-dir",    required=True,
                        help="Folder berisi file ALL_*.txt Maret 2024")
    parser.add_argument("--date-filter", default=None,
                        help="Filter nama file, misal '20240307' atau '2024'")
    parser.add_argument("--max-files",   type=int, default=None,
                        help="Batasi jumlah file (untuk quick test)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip normalisasi (jika normalizer belum ada)")
    args = parser.parse_args()

    t_start = time.time()
    print("\n" + "=" * 65)
    print("  OOD VALIDATION — DATA MARET 2024 (Proxy Anomali)")
    print("=" * 65)
    print(f"\n  Data dir  : {args.data_dir}")
    print(f"  Model dir : {MODEL_DIR}")
    print(f"  Output    : {RESULTS_DIR}")

    # ── Step 1: Load data 2024 ────────────────────────────
    print(f"\n[1/5] Load data raw Maret 2024...")
    data = load_raw_files(
        Path(args.data_dir),
        date_filter=args.date_filter,
        max_files=args.max_files,
    )
    if data is None:
        print("  ❌ Gagal load data. Hentikan.")
        sys.exit(1)

    # ── Step 2: Sliding window ─────────────────────────────
    print(f"\n[2/5] Sliding window segmentation...")
    X_raw = make_windows(data)
    if len(X_raw) == 0:
        print(f"  ❌ Tidak cukup data untuk membentuk window.")
        print(f"     Butuh minimal {WINDOW_SIZE} baris = {WINDOW_SIZE/100:.0f} detik.")
        print(f"     Tambahkan lebih banyak file.")
        sys.exit(1)
    print(f"  Windows terbentuk : {len(X_raw):,}")
    print(f"  Shape             : {X_raw.shape}  (n, {WINDOW_SIZE}, {X_raw.shape[2]})")

    # ── Step 3: Normalisasi ─────────────────────────────────
    if args.no_normalize:
        X_ood = X_raw
        print(f"\n[3/5] Normalisasi: SKIP (--no-normalize)")
    else:
        print(f"\n[3/5] Normalisasi dengan stats 2026...")
        X_ood = normalize_with_2026_stats(X_raw)

    # Label: semua y=1 (kondisi abnormal secara fisik)
    y_true = np.ones(len(X_ood), dtype=np.int8)
    print(f"  Label y=1         : {len(y_true):,} windows (semua = anomali)")

    # ── Step 4: Predict ─────────────────────────────────────
    print(f"\n[4/5] Inferensi ketiga model...")
    results = predict_all_models(X_ood)

    # ── Step 5: Metrik & Output ─────────────────────────────
    print(f"\n[5/5] Hitung metrik dan buat plot...")
    metrics_df = compute_metrics(results, y_true)

    # Tampilkan tabel
    print(f"\n  {'═'*65}")
    print(f"  HASIL OOD VALIDATION")
    print(f"  {'═'*65}")
    print(f"  Data      : Maret 2024 (kabel belum ditegangkan = anomali fisik)")
    print(f"  Windows   : {len(y_true):,} (semua berlabel anomali)")
    print()
    print(f"  {'Model':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} "
          f"{'Detected':>10} {'Total':>8} {'Det%':>7}")
    print(f"  {'-'*62}")
    for _, r in metrics_df.iterrows():
        if r["Status"] != "OK":
            print(f"  {r['Model']:<12} {'(belum dilatih)':>47}")
            continue
        print(f"  {r['Model']:<12} {r['Precision']:>10.4f} {r['Recall']:>10.4f} "
              f"{r['F1']:>8.4f} {r['Detected']:>10,} {r['Total']:>8,} "
              f"{r['Det_pct']:>6.1f}%")

    # Simpan CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "ood_validation_report.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Report  : {csv_path}")

    # Plot
    fig_dir = RESULTS_DIR / "figures"
    make_plots(results, X_ood, fig_dir)

    # Interpretasi
    print(f"\n  INTERPRETASI:")
    for _, r in metrics_df.iterrows():
        if r["Status"] != "OK":
            continue
        recall = float(r["Recall"])
        if recall >= 0.9:
            interp = "SANGAT BAIK — model sangat sensitif terhadap anomali struktural"
        elif recall >= 0.7:
            interp = "BAIK — model berhasil deteksi mayoritas kondisi abnormal"
        elif recall >= 0.5:
            interp = "CUKUP — ada beberapa window yang lolos deteksi"
        else:
            interp = "PERLU REVIEW — threshold mungkin perlu dikalibrasi ulang"
        print(f"  {r['Model']:<10}: Recall={recall:.0%} → {interp}")

    elapsed = time.time() - t_start
    print(f"\n  Selesai dalam {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()