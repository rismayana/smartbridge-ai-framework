"""
SHMS Bridge Anomaly Detection
shms_phase3a_lstm.py — LSTM Autoencoder

Prinsip kerja:
  - Dilatih HANYA dengan data normal (unsupervised)
  - Model belajar merekonstruksi pola time-series normal
  - Saat anomali: rekonstruksi buruk → reconstruction error (RE) tinggi
  - Threshold RE ditentukan dari distribusi data validation
  - Output: anomaly score per window (0.0 – 1.0)

Arsitektur:
  Input  (batch, 1000, 17)
    ↓ Encoder: LSTM stack → compressed representation
    ↓ Decoder: LSTM stack → rekonstruksi input
  Output (batch, 1000, 17)
  Loss   : MSE(input, output)

Cara pakai:
  python shms_phase3a_lstm.py --train       # training dari awal
  python shms_phase3a_lstm.py --eval        # evaluasi model tersimpan
  python shms_phase3a_lstm.py --train --colab  # mode Google Colab
"""

import argparse
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shms_config import (
    MODEL_DIR, RESULTS_DIR, DATA_PROCESSED_DIR, get_processed_dir,
    ALL_CHANNELS, MAIN_CHANNELS, CHANNEL_ALIAS,
    WINDOW_SIZE,
)
from shms_phase2_preprocessing import AdaptiveNormalizer

# ── Coba import PyTorch, fallback ke numpy jika tidak ada ──
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch tidak tersedia — mode simulasi aktif")
    print("       Install: pip install torch")


# ─────────────────────────────────────────────────────────
# HYPERPARAMETER — bisa di-tune
# ─────────────────────────────────────────────────────────

HP = {
    # Arsitektur
    "n_channels"       : len(MAIN_CHANNELS),  # 17
    "hidden_size"      : 64,
    "num_layers"       : 2,
    "dropout"          : 0.2,

    # Training — CPU mode, lazy-load dataset
    "batch_size"       : 64,          # dinaikkan dari 32 (lebih efisien per epoch)
    "learning_rate"    : 1e-3,
    "n_epochs"         : 100,
    "patience"         : 10,
    "clip_grad"        : 1.0,
    "max_train_windows": 50_000,      # subsample per epoch dari 411K total
                                       # hemat RAM + percepat epoch 8x
                                       # coverage penuh setelah ~8 epoch

    # Threshold
    "threshold_pct"    : 95,
}


# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class LSTMAutoencoder(nn.Module):
        """
        LSTM Autoencoder untuk anomaly detection time-series multi-channel.

        Encoder: kompres urutan 1000 timestep menjadi vektor laten
        Decoder: rekonstruksi urutan 1000 timestep dari vektor laten

        Input/Output shape: (batch, seq_len, n_channels)
        """

        def __init__(self, n_channels: int, hidden_size: int,
                     num_layers: int, dropout: float):
            super().__init__()
            self.n_channels  = n_channels
            self.hidden_size = hidden_size
            self.num_layers  = num_layers

            # Encoder: time-series → hidden representation
            self.encoder = nn.LSTM(
                input_size  = n_channels,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0,
            )

            # Decoder: hidden representation → rekonstruksi
            self.decoder = nn.LSTM(
                input_size  = hidden_size,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0,
            )

            # Proyeksi output ke dimensi channel asli
            self.output_layer = nn.Linear(hidden_size, n_channels)

        def forward(self, x):
            """
            x: (batch, seq_len, n_channels)
            returns: x_hat (batch, seq_len, n_channels) — rekonstruksi
            """
            batch_size, seq_len, _ = x.shape

            # Encode: proses seluruh sequence, ambil hidden state akhir
            _, (hidden, cell) = self.encoder(x)

            # Decode: gunakan hidden state encoder sebagai kondisi awal
            # Expand hidden state ke seluruh timestep
            decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
            decoder_out, _ = self.decoder(decoder_input, (hidden, cell))

            # Proyeksi ke channel asli
            x_hat = self.output_layer(decoder_out)
            return x_hat

        def reconstruction_error(self, x, x_hat):
            """
            Hitung per-window reconstruction error (MSE).
            Return shape: (batch,)
            """
            return ((x - x_hat) ** 2).mean(dim=(1, 2))


# ─────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────

class ShmsIterableDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset yang streaming 1 file .npy per saat.
    Tidak pakai mmap_mode → aman di Windows (path dengan spasi).
    Peak RAM: ~1.3 GB (1 hari) bukan 30 GB (semua hari sekaligus).

    Cara kerja per epoch:
    1. Shuffle urutan hari
    2. Per hari: load file .npy ke RAM (~1.3 GB)
    3. Subsample N_per_day windows secara acak
    4. Yield window satu per satu ke DataLoader
    5. Hapus dari RAM, lanjut hari berikutnya
    """
    def __init__(self, split: str, normal_only: bool = False,
                 windows_per_epoch: int = 50_000):
        self.split           = split
        self.normal_only     = normal_only
        self.windows_per_epoch = windows_per_epoch
        self.day_files       = []   # [(xp, yp), ...]

        summary_path = get_processed_dir() / "processing_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"processing_summary.csv tidak ditemukan: {summary_path}")

        summary = pd.read_csv(summary_path)
        days    = summary[
            (summary["split"] == split) & (summary["status"] == "ok")
        ]["date"].tolist()

        if not days:
            raise ValueError(f"Tidak ada data '{split}' dalam summary.")

        total_windows = 0
        for d in sorted(days):
            xp = get_processed_dir() / f"{d}_X.npy"
            yp = get_processed_dir() / f"{d}_y.npy"
            if xp.exists() and yp.exists():
                try:
                    # y.npy sangat kecil (<1MB) — aman di-load untuk cek n_rows
                    with open(str(yp), 'rb') as f:
                        y_tmp = np.load(f)
                    n_rows = len(y_tmp)
                    del y_tmp
                    self.day_files.append((str(xp), str(yp), n_rows))
                    total_windows += n_rows
                except Exception as e:
                    print(f"  [SKIP] {d}: {e}")

        self.total_windows = total_windows
        n_used = min(windows_per_epoch, total_windows) if windows_per_epoch else total_windows
        print(f"  [{split}] {total_windows:,} windows total "
              f"· {n_used:,}/epoch · {len(self.day_files)} hari")

    def __len__(self):
        """Estimasi panjang untuk DataLoader progress tracking."""
        if self.windows_per_epoch:
            return min(self.windows_per_epoch, self.total_windows)
        return self.total_windows

    def __iter__(self):
        import random, gc

        # Load normalizer SEKALI — verifikasi berhasil terbaca
        norm = None
        norm_path = get_processed_dir() / "p2_normalizer_stats.csv"
        if norm_path.exists():
            norm = AdaptiveNormalizer.load(str(norm_path))
            # Spot-check: pastikan cable tension ada di stats
            sample_ch = "FB_CA_L02_EZ"
            if sample_ch in norm.stats_:
                mu  = norm.stats_[sample_ch]['mean']
                sig = norm.stats_[sample_ch]['std']
                # Hanya print di iterasi pertama (epoch 1)
                if not hasattr(self, '_norm_verified'):
                    print(f"  [NORM OK] {sample_ch}: mu={mu:.1f}, sig={sig:.3f}")
                    self._norm_verified = True
        else:
            if not hasattr(self, '_norm_warned'):
                print(f"  [WARN] Normalizer tidak ditemukan: {norm_path}")
                self._norm_warned = True

        # Precompute ch_idx (index MAIN_CHANNELS dalam ALL_CHANNELS)
        ch_idx = [ALL_CHANNELS.index(c) for c in MAIN_CHANNELS
                  if c in ALL_CHANNELS]

        # Shuffle urutan hari — berbeda tiap epoch
        day_order = list(self.day_files)
        random.shuffle(day_order)

        n_target  = self.windows_per_epoch or self.total_windows
        collected = 0

        for xp, yp, n_rows in day_order:
            if collected >= n_target:
                break

            # Load via file object — aman di Windows (path spasi) & GDrive
            try:
                with open(xp, 'rb') as f:
                    X_day = np.load(f)
                with open(yp, 'rb') as f:
                    y_day = np.load(f)
            except Exception as e:
                print(f"  [SKIP] {Path(xp).stem}: {e}")
                continue

            # Filter normal-only
            if self.normal_only:
                X_day = X_day[y_day == 0]

            if len(X_day) == 0:
                del X_day, y_day; gc.collect()
                continue

            # Apply Z-score normalisasi (vectorized — tidak ada loop per channel)
            if norm is not None:
                for j, ch in enumerate(ALL_CHANNELS):
                    if ch in norm.stats_:
                        mu  = float(norm.stats_[ch]['mean'])
                        sig = float(norm.stats_[ch]['std'])
                        if sig > 0:
                            X_day[:, :, j] = (X_day[:, :, j] - mu) / sig

            # Pilih channel model: (n, 1000, 20) → (n, 1000, 17)
            X_main = X_day[:, :, ch_idx]   # view, bukan copy

            # Subsample proporsional
            n_from_day = min(n_target - collected, len(X_main))
            # Shuffle indices hari ini — berbeda tiap epoch
            idx = np.random.permutation(len(X_main))[:n_from_day]

            for i in idx:
                yield torch.FloatTensor(X_main[i].copy())
                collected += 1

            del X_day, y_day, X_main
            gc.collect()



def load_split_numpy(split: str, normal_only: bool = False,
                     max_windows: int = None) -> tuple:
    """
    Load data dari processed .npy files.
    normal_only=True → hanya ambil windows label=0 (untuk training)
    max_windows      → batasi jumlah windows (untuk quick test)
    """
    summary_path = get_processed_dir() / "processing_summary.csv"

    # Fallback: jika belum ada data processed, buat data simulasi
    if not summary_path.exists():
        print(f"  [SIMULASI] Data processed belum ada.")
        print(f"  Membuat data dummy untuk {split}...")
        n  = 1000 if split == "train" else 200
        X  = np.random.randn(n, WINDOW_SIZE, len(MAIN_CHANNELS)).astype(np.float32)
        y  = np.zeros(n, dtype=np.int8)
        if split != "train":
            # Tambahkan 10% anomali simulasi
            abn_idx = np.random.choice(n, n//10, replace=False)
            X[abn_idx] += np.random.randn(len(abn_idx), WINDOW_SIZE,
                                           len(MAIN_CHANNELS)) * 3
            y[abn_idx]  = 1
        return X, y

    summary = pd.read_csv(summary_path)
    days    = summary[
        (summary["split"] == split) & (summary["status"] == "ok")
    ]["date"].tolist()

    if not days:
        raise ValueError(f"Tidak ada data '{split}' yang sudah diproses.")

    X_list, y_list = [], []
    for d in sorted(days):
        xp = get_processed_dir() / f"{d}_X.npy"
        yp = get_processed_dir() / f"{d}_y.npy"
        if xp.exists():
            X_list.append(np.load(xp))
            y_list.append(np.load(yp))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if normal_only:
        mask = y == 0
        X, y = X[mask], y[mask]

    if max_windows and len(X) > max_windows:
        idx  = np.random.choice(len(X), max_windows, replace=False)
        idx  = np.sort(idx)
        X, y = X[idx], y[idx]

    print(f"  [{split}] {len(X):,} windows "
          f"({y.sum()} abnormal, {100*y.mean():.1f}%)")
    return X.astype(np.float32), y


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────

def train(hp: dict = HP, save_dir: Path = MODEL_DIR,
          results_dir: Path = RESULTS_DIR):
    """
    Training LSTM Autoencoder.
    - Hanya pakai data normal untuk training
    - Early stopping berdasarkan validation loss
    - Simpan model terbaik (val loss terendah)
    """
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch diperlukan untuk training.")
        print("        Install: pip install torch")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  LSTM AUTOENCODER — TRAINING")
    print(f"  Device : {device}")
    print(f"  Channels: {hp['n_channels']} | Hidden: {hp['hidden_size']} "
          f"| Layers: {hp['num_layers']}")
    print(f"  Epochs : {hp['n_epochs']} | Batch: {hp['batch_size']} "
          f"| LR: {hp['learning_rate']}")
    print(f"{'='*65}")

    # Load data — lazy dataset (tidak load semua ke RAM)
    print("\n[DATA]")
    train_ds = ShmsIterableDataset("train", normal_only=True)
    val_ds   = ShmsIterableDataset("val",   normal_only=True)

    # IterableDataset: subsample sudah dilakukan di dalam __iter__
    # Tidak perlu SubsetRandomSampler — DataLoader langsung streaming
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"],
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=hp["batch_size"],
                          num_workers=0, pin_memory=False)

    # Model, loss, optimizer
    model     = LSTMAutoencoder(
        n_channels  = hp["n_channels"],
        hidden_size = hp["hidden_size"],
        num_layers  = hp["num_layers"],
        dropout     = hp["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Training loop
    print(f"\n[TRAINING] {len(train_ds):,} train | {len(val_ds):,} val windows")
    history       = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, hp["n_epochs"] + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch_x in train_dl:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat = model(batch_x)
            loss  = criterion(x_hat, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip_grad"])
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_dl:
                batch_x = batch_x.to(device)
                x_hat   = model(batch_x)
                loss    = criterion(x_hat, batch_x)
                val_loss += loss.item() * len(batch_x)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{hp['n_epochs']} | "
              f"train={train_loss:.6f} | val={val_loss:.6f} | "
              f"{elapsed:.1f}s", end="")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_cnt  = 0
            print(" ✓ best")
        else:
            patience_cnt += 1
            print(f" (patience {patience_cnt}/{hp['patience']})")
            if patience_cnt >= hp["patience"]:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Restore best model
    model.load_state_dict(best_state)

    # Simpan model
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / "lstm_autoencoder_best.pt"
    torch.save({
        "model_state": best_state,
        "hp"         : hp,
        "best_val_loss": best_val_loss,
        "n_epochs_trained": epoch,
    }, model_path)

    # Simpan hyperparameter
    hp_path = save_dir / "lstm_hp.json"
    with open(hp_path, "w") as f:
        json.dump({**hp, "best_val_loss": best_val_loss,
                   "epochs_trained": epoch}, f, indent=2)

    print(f"\n  Model disimpan: {model_path}")

    # Plot loss curve
    _plot_loss(history, results_dir / "figures" / "lstm_loss_curve.png")

    return model, history


# ─────────────────────────────────────────────────────────
# THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────

def calibrate_threshold(model, hp: dict = HP,
                        save_dir: Path = MODEL_DIR,
                        results_dir: Path = RESULTS_DIR):
    """
    Hitung threshold reconstruction error dari data validation.
    Threshold = persentil ke-N dari RE distribusi normal.

    Ini adalah langkah kritis:
    - Terlalu rendah → banyak false positive
    - Terlalu tinggi → banyak false negative
    """
    if not TORCH_AVAILABLE:
        return None

    device = next(model.parameters()).device
    print(f"\n[THRESHOLD] Kalibrasi dari data validation...")

    val_ds = ShmsIterableDataset("val", normal_only=False)
    val_dl = DataLoader(val_ds, batch_size=hp["batch_size"],
                        shuffle=False, num_workers=0)

    model.eval()
    all_re = []
    with torch.no_grad():
        for batch_x in val_dl:
            batch_x = batch_x.to(device)
            x_hat   = model(batch_x)
            re      = model.reconstruction_error(batch_x, x_hat)
            all_re.extend(re.cpu().numpy())

    all_re    = np.array(all_re)
    y_val = np.array([0] * len(val_ds))   # semua normal (y=0 dari 2026)
    re_normal = all_re[y_val == 0]

    threshold = np.percentile(re_normal, hp["threshold_pct"])

    print(f"  RE normal  : mean={re_normal.mean():.6f}, "
          f"std={re_normal.std():.6f}")
    print(f"  Threshold  : {threshold:.6f} "
          f"(persentil {hp['threshold_pct']})")

    # Simpan threshold
    thresh_path = save_dir / "lstm_threshold.json"
    with open(thresh_path, "w") as f:
        json.dump({
            "threshold"    : float(threshold),
            "threshold_pct": hp["threshold_pct"],
            "re_mean_normal": float(re_normal.mean()),
            "re_std_normal" : float(re_normal.std()),
        }, f, indent=2)

    # Plot distribusi RE
    _plot_re_distribution(
        all_re, y_val, threshold,
        results_dir / "figures" / "lstm_re_distribution.png"
    )

    return threshold


# ─────────────────────────────────────────────────────────
# EVALUASI
# ─────────────────────────────────────────────────────────

def evaluate(model=None, threshold: float = None,
             hp: dict = HP,
             save_dir: Path = MODEL_DIR,
             results_dir: Path = RESULTS_DIR) -> dict:
    """
    Evaluasi model pada data test.
    Hitung: precision, recall, F1, AUC-ROC.
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
        classification_report
    )

    if not TORCH_AVAILABLE:
        # Mode simulasi
        print("\n[EVAL SIMULASI] PyTorch tidak tersedia")
        print("  Menampilkan contoh format output evaluasi...")
        metrics = {
            "model"    : "LSTM_Autoencoder",
            "precision": 0.0, "recall": 0.0,
            "f1"       : 0.0, "auc"   : 0.0,
            "note"     : "simulasi — install PyTorch untuk hasil nyata"
        }
        return metrics

    device = next(model.parameters()).device

    # Load threshold jika tidak diberikan
    if threshold is None:
        thresh_path = save_dir / "lstm_threshold.json"
        if thresh_path.exists():
            with open(thresh_path) as f:
                threshold = json.load(f)["threshold"]
        else:
            raise ValueError("Threshold belum dikalibrasi. Jalankan calibrate_threshold() dulu.")

    print(f"\n[EVALUASI] Data test | Threshold: {threshold:.6f}")

    test_ds  = ShmsIterableDataset("test", normal_only=False)
    X_test   = None   # tidak load ke RAM, gunakan DataLoader
    y_test   = np.zeros(len(test_ds), dtype=np.int8)   # semua normal
    test_ds = TensorDataset(torch.FloatTensor(X_test))
    test_dl = DataLoader(test_ds, batch_size=hp["batch_size"],
                         shuffle=False, num_workers=0)

    model.eval()
    all_re = []
    with torch.no_grad():
        for batch_x in test_dl:
            batch_x = batch_x.to(device)
            x_hat   = model(batch_x)
            re      = model.reconstruction_error(batch_x, x_hat)
            all_re.extend(re.cpu().numpy())

    all_re  = np.array(all_re)
    y_pred  = (all_re > threshold).astype(int)

    # Normalisasi RE ke [0, 1] sebagai anomaly score
    re_min, re_max = all_re.min(), all_re.max()
    scores = (all_re - re_min) / (re_max - re_min + 1e-9)

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, scores) if y_test.sum() > 0 else 0.0
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:5,}  FP={cm[0,1]:5,}")
    print(f"    FN={cm[1,0]:5,}  TP={cm[1,1]:5,}")
    print(f"\n  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    metrics = {
        "model"    : "LSTM_Autoencoder",
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
        "auc"      : round(auc,       4),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        "n_test_windows"   : len(y_test),
        "n_test_abnormal"  : int(y_test.sum()),
    }

    # Simpan ke results
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        results_dir / "lstm_metrics.csv", index=False
    )

    # Update experiment log
    _update_experiment_log(metrics, results_dir.parent)

    # Plot hasil
    _plot_anomaly_timeline(
        all_re, y_test, threshold, scores,
        results_dir / "figures" / "lstm_anomaly_timeline.png"
    )
    _plot_confusion_matrix(
        cm, results_dir / "figures" / "lstm_confusion_matrix.png"
    )

    return metrics


# ─────────────────────────────────────────────────────────
# INFERENCE — deteksi anomali pada data baru
# ─────────────────────────────────────────────────────────

def predict(X: np.ndarray, model=None, threshold: float = None,
            save_dir: Path = MODEL_DIR) -> dict:
    """
    Deteksi anomali pada data baru.
    X: (n_windows, WINDOW_SIZE, n_channels) — sudah dinormalisasi

    Return: dict dengan scores dan predictions
    """
    if not TORCH_AVAILABLE:
        return {"scores": np.zeros(len(X)), "predictions": np.zeros(len(X))}

    if model is None:
        model, threshold = load_model(save_dir)

    device = next(model.parameters()).device
    ds = TensorDataset(torch.FloatTensor(X))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    all_re = []
    with torch.no_grad():
        for batch_x in dl:
            batch_x = batch_x.to(device)
            x_hat   = model(batch_x)
            re      = model.reconstruction_error(batch_x, x_hat)
            all_re.extend(re.cpu().numpy())

    all_re  = np.array(all_re)
    re_min, re_max = all_re.min(), all_re.max()
    scores  = (all_re - re_min) / (re_max - re_min + 1e-9)
    preds   = (all_re > threshold).astype(int)

    return {
        "reconstruction_error": all_re,
        "scores"     : scores,
        "predictions": preds,
        "threshold"  : threshold,
    }


def load_model(save_dir: Path = MODEL_DIR):
    """Load model dan threshold dari file tersimpan."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch diperlukan.")

    model_path = save_dir / "lstm_autoencoder_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    ckpt  = torch.load(model_path, map_location="cpu")
    hp    = ckpt["hp"]
    model = LSTMAutoencoder(
        n_channels  = hp["n_channels"],
        hidden_size = hp["hidden_size"],
        num_layers  = hp["num_layers"],
        dropout     = hp["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    thresh_path = save_dir / "lstm_threshold.json"
    threshold   = None
    if thresh_path.exists():
        with open(thresh_path) as f:
            threshold = json.load(f)["threshold"]

    print(f"  Model loaded: val_loss={ckpt['best_val_loss']:.6f}, "
          f"threshold={threshold}")
    return model, threshold


# ─────────────────────────────────────────────────────────
# HELPER PLOTS
# ─────────────────────────────────────────────────────────

def _plot_loss(history: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss", color="#378ADD", lw=1.5)
    ax.plot(history["val_loss"],   label="Val loss",   color="#D85A30", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Autoencoder — Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path.name}")


def _plot_re_distribution(all_re, y_true, threshold, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    re_normal = all_re[y_true == 0]
    re_abnorm = all_re[y_true == 1]
    ax.hist(re_normal, bins=60, alpha=0.7, color="#378ADD", label="Normal")
    if len(re_abnorm):
        ax.hist(re_abnorm, bins=60, alpha=0.7, color="#E24B4A", label="Abnormal")
    ax.axvline(threshold, color="#BA7517", lw=2, linestyle="--",
               label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("LSTM — Distribusi Reconstruction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path.name}")


def _plot_anomaly_timeline(all_re, y_true, threshold, scores, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(all_re, color="#888780", lw=0.6, alpha=0.8, label="RE")
    axes[0].axhline(threshold, color="#BA7517", lw=1.5, linestyle="--",
                    label=f"Threshold={threshold:.4f}")
    if y_true.sum():
        abn_idx = np.where(y_true == 1)[0]
        axes[0].scatter(abn_idx, all_re[abn_idx], color="#E24B4A",
                        s=8, zorder=5, label="Actual abnormal")
    axes[0].set_ylabel("Reconstruction Error")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    axes[1].fill_between(range(len(scores)), scores, alpha=0.6,
                         color="#378ADD", label="Anomaly score")
    axes[1].axhline(0.5, color="#BA7517", lw=1, linestyle="--")
    axes[1].set_ylabel("Anomaly Score (0–1)")
    axes[1].set_xlabel("Window index")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("LSTM Autoencoder — Anomaly Detection Timeline", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path.name}")


def _plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Normal", "Abnormal"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — LSTM Autoencoder")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path.name}")


def _update_experiment_log(metrics: dict, project_root: Path):
    """Catat hasil ke experiment_log.csv."""
    log_path = project_root / "05_results" / "experiment_log.csv"
    row = {
        "run_id"   : pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": pd.Timestamp.now().isoformat(),
        "model"    : metrics.get("model"),
        "window_size": WINDOW_SIZE,
        "threshold": metrics.get("threshold"),
        "precision": metrics.get("precision"),
        "recall"   : metrics.get("recall"),
        "f1"       : metrics.get("f1"),
        "auc"      : metrics.get("auc"),
        "notes"    : "",
    }
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def run_phase3a(retrain: bool = False) -> dict:
    """
    Jalankan Phase 3A lengkap.
    Dipanggil dari run_pipeline.py atau langsung.
    """
    print(f"\n{'='*65}")
    print(f"  PHASE 3A — LSTM AUTOENCODER")
    print(f"{'='*65}")

    model_path = MODEL_DIR / "lstm_autoencoder_best.pt"

    if not model_path.exists() or retrain:
        # Training dari awal
        model, history = train(HP, MODEL_DIR, RESULTS_DIR)
        if model is None:
            return {}
        threshold = calibrate_threshold(model, HP, MODEL_DIR, RESULTS_DIR)
    else:
        # Load model yang sudah ada
        print(f"  Model sudah ada, load dari: {model_path}")
        model, threshold = load_model(MODEL_DIR)

    metrics = evaluate(model, threshold, HP, MODEL_DIR, RESULTS_DIR)

    print(f"\n  PHASE 3A SELESAI")
    print(f"  F1={metrics.get('f1', 0):.4f} | "
          f"AUC={metrics.get('auc', 0):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="SHMS LSTM Autoencoder"
    )
    parser.add_argument("--train",  action="store_true",
                        help="Training model dari awal")
    parser.add_argument("--eval",   action="store_true",
                        help="Evaluasi model tersimpan")
    parser.add_argument("--retrain",action="store_true",
                        help="Retrain meski model sudah ada")
    args = parser.parse_args()

    if args.train or args.retrain:
        model, history = train(HP, MODEL_DIR, RESULTS_DIR)
        if model:
            threshold = calibrate_threshold(model, HP, MODEL_DIR, RESULTS_DIR)
            evaluate(model, threshold, HP, MODEL_DIR, RESULTS_DIR)
    elif args.eval:
        model, threshold = load_model(MODEL_DIR)
        evaluate(model, threshold, HP, MODEL_DIR, RESULTS_DIR)
    else:
        # Default: jalankan full pipeline
        run_phase3a()


if __name__ == "__main__":
    main()
