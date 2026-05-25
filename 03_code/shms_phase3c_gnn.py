"""
SHMS Bridge Anomaly Detection
shms_phase3c_gnn.py — Graph Neural Network (GNN)

Prinsip kerja:
  - Setiap sensor = node dalam graph
  - Edge antar node = korelasi fisik antar sensor (dari Phase 2)
  - GNN belajar pola hubungan normal antar sensor
  - Anomali = pola hubungan menyimpang dari yang dipelajari

Novelty untuk paper:
  Berbeda dari LSTM (temporal) dan IForest (statistik per sensor),
  GNN mendeteksi anomali dari PERUBAHAN KORELASI SPASIAL antar sensor.
  Contoh: accelerometer PY1T dan cable L22 biasanya berkorelasi kuat.
  Jika tiba-tiba korelasi melemah → GNN mendeteksi anomali struktural.

Arsitektur:
  Input: node features (window features per sensor) + adjacency matrix
    ↓ GCN Layer 1: agregasi informasi dari tetangga
    ↓ GCN Layer 2: representasi high-level per node
    ↓ Global pooling: representasi seluruh graph
    ↓ Decoder: rekonstruksi node features
  Loss: MSE(input features, rekonstruksi)

Cara pakai:
  python shms_phase3c_gnn.py --train
  python shms_phase3c_gnn.py --eval
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from shms_config import (
    MODEL_DIR, RESULTS_DIR, DATA_PROCESSED_DIR,
    MAIN_CHANNELS, CHANNEL_ALIAS, WINDOW_SIZE,
)

# ── Coba import PyTorch & PyG ──────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ─────────────────────────────────────────────────────────
# HYPERPARAMETER
# ─────────────────────────────────────────────────────────

HP = {
    # Graph construction
    "corr_threshold"  : 0.5,    # edge jika |korelasi| >= nilai ini
    "node_feat_size"  : 8,      # fitur per node (8 statistik: mean,std,dll)

    # Arsitektur GNN
    "hidden_channels" : 32,     # ukuran hidden layer GCN
    "n_gcn_layers"    : 2,      # jumlah GCN layer
    "dropout"         : 0.2,

    # Training
    "batch_size"      : 64,
    "learning_rate"   : 1e-3,
    "n_epochs"        : 50,
    "patience"        : 7,
    "clip_grad"       : 1.0,

    # Threshold
    "threshold_pct"   : 95,
}

N_NODES = len(MAIN_CHANNELS)   # 17 sensor = 17 node


# ─────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────

def build_adjacency(corr_matrix: np.ndarray,
                    threshold: float = 0.5) -> tuple:
    """
    Bangun adjacency dari correlation matrix.

    Returns:
        edge_index : (2, n_edges) — pasangan node yang terhubung
        edge_weight: (n_edges,)   — bobot edge (nilai korelasi)
        adj_matrix : (N, N)       — adjacency matrix
    """
    n = corr_matrix.shape[0]
    adj = np.abs(corr_matrix)
    np.fill_diagonal(adj, 0)   # hapus self-loop

    rows, cols = np.where(adj >= threshold)
    edge_index  = np.array([rows, cols])
    edge_weight = adj[rows, cols]

    # Pastikan ada minimal self-loop jika tidak ada edge
    if len(rows) == 0:
        edge_index  = np.array([np.arange(n), np.arange(n)])
        edge_weight = np.ones(n)

    return edge_index, edge_weight, adj


def load_correlation_matrix() -> np.ndarray:
    """
    Load correlation matrix dari Phase 2.
    Fallback ke identity matrix jika belum ada.
    """
    corr_path = DATA_PROCESSED_DIR / "p2_correlation_matrix.csv"
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path, index_col=0)
        corr    = corr_df.values.astype(np.float32)
        print(f"  Correlation matrix loaded: {corr.shape}")
        return corr

    # Fallback: buat matrix dengan korelasi moderat
    print("  [WARN] Correlation matrix belum ada, pakai default (|corr|=0.3)")
    n    = N_NODES
    corr = np.full((n, n), 0.3, dtype=np.float32)
    np.fill_diagonal(corr, 1.0)
    # Tambahkan korelasi tinggi antara sensor yang dekat secara fisik
    # Accelerometer di lokasi sama: PY1T (0,1), PY1D (2,3), S1Q1 (5,6), S3Q3 (7,8)
    for pair in [(0,1),(2,3),(5,6),(7,8),(9,10),(11,12),(13,14)]:
        corr[pair[0], pair[1]] = 0.8
        corr[pair[1], pair[0]] = 0.8
    return corr


def extract_node_features(X: np.ndarray) -> np.ndarray:
    """
    Ekstrak fitur per node (sensor) dari window.

    Input  X: (n_windows, window_size, n_channels)
    Output  : (n_windows, n_nodes, node_feat_size)
              8 fitur statistik per node per window
    """
    n_windows, win_size, n_ch = X.shape
    n_feat = HP["node_feat_size"]   # 8
    out = np.zeros((n_windows, n_ch, n_feat), dtype=np.float32)

    for i in range(n_windows):
        for j in range(n_ch):
            s = X[i, :, j]
            s_clean = s[~np.isnan(s)]
            if len(s_clean) < 2:
                continue
            out[i, j, 0] = s_clean.mean()
            out[i, j, 1] = s_clean.std()
            out[i, j, 2] = s_clean.min()
            out[i, j, 3] = s_clean.max()
            out[i, j, 4] = s_clean.max() - s_clean.min()
            out[i, j, 5] = np.sqrt((s_clean**2).mean())

            # ── PERBAIKAN BUG NaN ────────────────────────────────────────
            # stats.skew() dan stats.kurtosis() mengembalikan NaN jika
            # sinyal konstan (std = 0). Ini terjadi saat channel cable
            # tension Nov 2025 dinormalisasi dengan stats training 2026:
            # sinyal flat menghasilkan z-score konstan → std lokal = 0.
            # Satu NaN di fitur node cukup untuk meracuni seluruh output
            # GCN melalui message passing (NaN × W → NaN di semua node).
            # Solusi: fallback ke 0.0 saat std = 0 (distribusi degenerate).
            if s_clean.std() < 1e-8:
                out[i, j, 6] = 0.0   # skewness = 0 (distribusi simetris)
                out[i, j, 7] = 0.0   # kurtosis = 0 (distribusi normal excess)
            else:
                sk = float(stats.skew(s_clean))
                ku = float(stats.kurtosis(s_clean))
                out[i, j, 6] = 0.0 if np.isnan(sk) else sk
                out[i, j, 7] = 0.0 if np.isnan(ku) else ku
            # ─────────────────────────────────────────────────────────────

    return out


# ─────────────────────────────────────────────────────────
# GNN MODEL (PyTorch Geometric)
# ─────────────────────────────────────────────────────────

if TORCH_AVAILABLE and PYG_AVAILABLE:

    class GNNAutoencoder(nn.Module):
        """
        Graph Autoencoder untuk anomaly detection.

        Encoder: node features → compressed graph representation
        Decoder: rekonstruksi node features dari graph representation

        Anomali dideteksi jika rekonstruksi error tinggi
        → pola korelasi antar sensor menyimpang dari normal.
        """

        def __init__(self, node_feat_size: int, hidden_channels: int,
                     n_layers: int, dropout: float):
            super().__init__()
            self.n_layers = n_layers

            # Encoder: stack GCN layers
            self.enc_convs = nn.ModuleList()
            in_ch = node_feat_size
            for i in range(n_layers):
                out_ch = hidden_channels if i < n_layers - 1 else hidden_channels // 2
                self.enc_convs.append(GCNConv(in_ch, out_ch))
                in_ch = out_ch

            # Bottleneck size
            bottleneck = hidden_channels // 2

            # Decoder: rekonstruksi node features
            self.dec_layers = nn.ModuleList()
            in_ch = bottleneck
            for i in range(n_layers):
                out_ch = hidden_channels if i < n_layers - 1 else node_feat_size
                self.dec_layers.append(nn.Linear(in_ch, out_ch))
                in_ch = out_ch

            self.dropout = dropout

        def encode(self, x, edge_index, edge_weight=None):
            for i, conv in enumerate(self.enc_convs):
                x = conv(x, edge_index, edge_weight)
                if i < self.n_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def decode(self, z):
            for i, layer in enumerate(self.dec_layers):
                z = layer(z)
                if i < self.n_layers - 1:
                    z = F.relu(z)
            return z

        def forward(self, x, edge_index, edge_weight=None):
            z     = self.encode(x, edge_index, edge_weight)
            x_hat = self.decode(z)
            return x_hat

        def reconstruction_error(self, x, x_hat):
            """Per-graph MSE. Shape: (1,)"""
            return ((x - x_hat) ** 2).mean()


# ─────────────────────────────────────────────────────────
# GNN TANPA PyG — implementasi manual menggunakan numpy/torch
# (fallback jika torch-geometric tidak tersedia)
# ─────────────────────────────────────────────────────────

if TORCH_AVAILABLE and not PYG_AVAILABLE:

    class SimpleGCNLayer(nn.Module):
        """GCN layer sederhana tanpa PyG."""
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x, adj):
            # Normalisasi adjacency: D^-1/2 A D^-1/2
            deg  = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            norm = deg ** -0.5
            adj_norm = norm * adj * norm.transpose(-1, -2)
            # Agregasi: AXW
            agg  = torch.bmm(adj_norm, x)
            return F.relu(self.linear(agg))


    class GNNAutoencoder(nn.Module):
        """GNN Autoencoder tanpa PyG dependency."""

        def __init__(self, node_feat_size, hidden_channels,
                     n_layers, dropout):
            super().__init__()
            self.encoder = nn.Sequential(
                SimpleGCNLayer(node_feat_size, hidden_channels),
                SimpleGCNLayer(hidden_channels, hidden_channels // 2),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels // 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, node_feat_size),
            )
            self.dropout_p = dropout

        def forward(self, x, adj, edge_weight=None):
            # x: (batch, n_nodes, node_feat)
            # adj: (n_nodes, n_nodes) broadcast ke batch
            b = x.shape[0]
            adj_batch = adj.unsqueeze(0).expand(b, -1, -1)
            z = x
            for layer in self.encoder:
                z = layer(z, adj_batch)
                z = F.dropout(z, p=self.dropout_p, training=self.training)
            x_hat = self.decoder(z)
            return x_hat

        def reconstruction_error(self, x, x_hat):
            return ((x - x_hat)**2).mean(dim=(1,2))


# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────

def load_split(split: str, normal_only: bool = False,
               max_windows: int = None) -> tuple:
    """Load data .npy dengan fallback simulasi."""
    summary_path = DATA_PROCESSED_DIR / "processing_summary.csv"

    if not summary_path.exists():
        print(f"  [SIMULASI] Data dummy untuk '{split}'...")
        n = {"train": 2000, "val": 400, "test": 400}[split]
        X = np.random.randn(n, WINDOW_SIZE,
                             N_NODES).astype(np.float32)
        y = np.zeros(n, dtype=np.int8)
        if split != "train":
            abn    = np.random.choice(n, n // 10, replace=False)
            X[abn] += (np.random.randn(len(abn), WINDOW_SIZE,
                                        N_NODES) * 3).astype(np.float32)
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

    X = np.concatenate(X_list).astype(np.float32)
    y = np.concatenate(y_list)

    if normal_only:
        X, y = X[y == 0], y[y == 0]
    if max_windows and len(X) > max_windows:
        idx  = np.sort(np.random.choice(len(X), max_windows, replace=False))
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
    """Training GNN Autoencoder."""

    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch diperlukan. Install: pip install torch")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  GNN AUTOENCODER — TRAINING")
    print(f"  Device : {device}")
    print(f"  Nodes  : {N_NODES} sensor | "
          f"Node feat: {hp['node_feat_size']} | "
          f"Hidden: {hp['hidden_channels']}")
    print(f"  PyG    : {'tersedia' if PYG_AVAILABLE else 'tidak tersedia — pakai fallback'}")
    print(f"{'='*65}")

    # Load data dan adjacency
    print("\n[GRAPH]")
    corr        = load_correlation_matrix()
    edge_index, edge_weight, adj = build_adjacency(
        corr, hp["corr_threshold"]
    )
    n_edges = edge_index.shape[1]
    print(f"  Nodes : {N_NODES} | Edges: {n_edges} "
          f"(threshold={hp['corr_threshold']})")

    # Visualisasi graph
    _plot_graph(corr, adj, hp["corr_threshold"],
                results_dir / "figures" / "gnn_graph_structure.png")

    # Load data
    print("\n[DATA]")
    X_train, _ = load_split("train", normal_only=True)
    X_val,   _ = load_split("val",   normal_only=True)

    # Ekstrak node features
    print("\n[FEATURES]")
    print(f"  Mengekstrak node features dari {len(X_train):,} windows...")
    NF_train = extract_node_features(X_train)
    NF_val   = extract_node_features(X_val)
    print(f"  Shape: {NF_train.shape} "
          f"(windows, nodes={N_NODES}, feats={hp['node_feat_size']})")

    # Model
    model = GNNAutoencoder(
        node_feat_size  = hp["node_feat_size"],
        hidden_channels = hp["hidden_channels"],
        n_layers        = hp["n_gcn_layers"],
        dropout         = hp["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Siapkan tensors
    if PYG_AVAILABLE:
        ei  = torch.LongTensor(edge_index).to(device)
        ew  = torch.FloatTensor(edge_weight).to(device)
    else:
        adj_t = torch.FloatTensor(adj).to(device)

    NF_train_t = torch.FloatTensor(NF_train)
    NF_val_t   = torch.FloatTensor(NF_val)

    # DataLoader
    from torch.utils.data import TensorDataset
    train_dl = DataLoader(TensorDataset(NF_train_t),
                          batch_size=hp["batch_size"],
                          shuffle=True, num_workers=0)
    val_dl   = DataLoader(TensorDataset(NF_val_t),
                          batch_size=hp["batch_size"],
                          shuffle=False, num_workers=0)

    # Training loop
    print(f"\n[TRAINING]")
    history      = {"train_loss": [], "val_loss": []}
    best_val     = float("inf")
    patience_cnt = 0
    best_state   = None
    import time

    for epoch in range(1, hp["n_epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for (bx,) in train_dl:
            bx = bx.to(device)
            optimizer.zero_grad()
            if PYG_AVAILABLE:
                # Proses tiap graph dalam batch
                losses = []
                for i in range(len(bx)):
                    x_i   = bx[i]
                    x_hat = model(x_i, ei, ew)
                    losses.append(((x_i - x_hat)**2).mean())
                loss = torch.stack(losses).mean()
            else:
                x_hat = model(bx, adj_t)
                loss  = ((bx - x_hat)**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), hp["clip_grad"])
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(NF_train)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (bx,) in val_dl:
                bx = bx.to(device)
                if PYG_AVAILABLE:
                    losses = []
                    for i in range(len(bx)):
                        x_i   = bx[i]
                        x_hat = model(x_i, ei, ew)
                        losses.append(((x_i - x_hat)**2).mean())
                    loss = torch.stack(losses).mean()
                else:
                    x_hat = model(bx, adj_t)
                    loss  = ((bx - x_hat)**2).mean()
                val_loss += loss.item() * len(bx)
        val_loss /= len(NF_val)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        mark = ""
        if val_loss < best_val:
            best_val     = val_loss
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_cnt = 0
            mark = " ✓ best"
        else:
            patience_cnt += 1

        print(f"  Epoch {epoch:3d}/{hp['n_epochs']} | "
              f"train={train_loss:.6f} | val={val_loss:.6f} | "
              f"{elapsed:.1f}s{mark}")

        if patience_cnt >= hp["patience"]:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # Simpan
    save_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state"    : best_state,
        "hp"             : hp,
        "best_val_loss"  : best_val,
        "edge_index"     : edge_index.tolist(),
        "edge_weight"    : edge_weight.tolist(),
        "adj_matrix"     : adj.tolist(),
        "n_epochs_trained": epoch,
    }, save_dir / "gnn_model_best.pt")

    with open(save_dir / "gnn_hp.json", "w") as f:
        json.dump({**hp, "best_val_loss": best_val,
                   "epochs": epoch}, f, indent=2)

    _plot_loss(history, results_dir / "figures" / "gnn_loss_curve.png")

    print(f"\n  Model disimpan: {save_dir/'gnn_model_best.pt'}")
    return model, (edge_index, edge_weight, adj)


# ─────────────────────────────────────────────────────────
# THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────

def calibrate_threshold(model, graph_data: tuple,
                        hp: dict = HP,
                        save_dir: Path = MODEL_DIR,
                        results_dir: Path = RESULTS_DIR) -> float:
    """Kalibrasi threshold dari validation set."""
    if not TORCH_AVAILABLE:
        return None

    device     = next(model.parameters()).device
    edge_index, edge_weight, adj = graph_data

    if PYG_AVAILABLE:
        ei  = torch.LongTensor(edge_index).to(device)
        ew  = torch.FloatTensor(edge_weight).to(device)
    else:
        adj_t = torch.FloatTensor(adj).to(device)

    print("\n[THRESHOLD]")
    X_val, y_val = load_split("val", normal_only=False)
    NF_val       = extract_node_features(X_val)

    model.eval()
    all_re = []
    with torch.no_grad():
        for i in range(0, len(NF_val), hp["batch_size"]):
            batch = torch.FloatTensor(
                NF_val[i:i+hp["batch_size"]]).to(device)
            if PYG_AVAILABLE:
                errors = []
                for j in range(len(batch)):
                    x_hat = model(batch[j], ei, ew)
                    errors.append(((batch[j]-x_hat)**2).mean().item())
                all_re.extend(errors)
            else:
                x_hat = model(batch, adj_t)
                re    = ((batch - x_hat)**2).mean(dim=(1,2))
                all_re.extend(re.cpu().numpy())

    all_re    = np.array(all_re)
    re_normal = all_re[y_val == 0]
    threshold = np.percentile(re_normal, hp["threshold_pct"])

    print(f"  RE normal: mean={re_normal.mean():.6f}, "
          f"std={re_normal.std():.6f}")
    print(f"  Threshold: {threshold:.6f} (P{hp['threshold_pct']})")

    with open(save_dir / "gnn_threshold.json", "w") as f:
        json.dump({
            "threshold"      : float(threshold),
            "threshold_pct"  : hp["threshold_pct"],
            "re_mean_normal" : float(re_normal.mean()),
            "re_std_normal"  : float(re_normal.std()),
        }, f, indent=2)

    _plot_re_distribution(
        all_re, y_val, threshold,
        results_dir / "figures" / "gnn_re_distribution.png"
    )

    return threshold


# ─────────────────────────────────────────────────────────
# EVALUASI
# ─────────────────────────────────────────────────────────

def evaluate(model=None, graph_data=None, threshold=None,
             hp: dict = HP,
             save_dir: Path = MODEL_DIR,
             results_dir: Path = RESULTS_DIR) -> dict:
    """Evaluasi pada data test."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    if not TORCH_AVAILABLE:
        return {}

    if model is None:
        model, graph_data, threshold = load_model(save_dir)

    device     = next(model.parameters()).device
    edge_index, edge_weight, adj = graph_data

    if PYG_AVAILABLE:
        ei  = torch.LongTensor(edge_index).to(device)
        ew  = torch.FloatTensor(edge_weight).to(device)
    else:
        adj_t = torch.FloatTensor(adj).to(device)

    print(f"\n[EVALUASI] Test set | Threshold: {threshold:.6f}")
    X_test, y_test = load_split("test", normal_only=False)
    NF_test        = extract_node_features(X_test)

    model.eval()
    all_re = []
    with torch.no_grad():
        for i in range(0, len(NF_test), hp["batch_size"]):
            batch = torch.FloatTensor(
                NF_test[i:i+hp["batch_size"]]).to(device)
            if PYG_AVAILABLE:
                errors = []
                for j in range(len(batch)):
                    x_hat = model(batch[j], ei, ew)
                    errors.append(((batch[j]-x_hat)**2).mean().item())
                all_re.extend(errors)
            else:
                x_hat = model(batch, adj_t)
                re    = ((batch - x_hat)**2).mean(dim=(1,2))
                all_re.extend(re.cpu().numpy())

    all_re  = np.array(all_re)
    re_min, re_max = all_re.min(), all_re.max()
    scores  = (all_re - re_min) / (re_max - re_min + 1e-9)
    y_pred  = (all_re > threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, scores) if y_test.sum() > 0 else 0.0
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:6,}  FP={cm[0,1]:6,}")
    print(f"    FN={cm[1,0]:6,}  TP={cm[1,1]:6,}")
    print(f"\n  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    metrics = {
        "model"    : "GNN_Autoencoder",
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
        "auc"      : round(auc,       4),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        "n_test_windows"  : len(y_test),
        "n_test_abnormal" : int(y_test.sum()),
        "n_nodes"         : N_NODES,
        "n_edges"         : edge_index.shape[1],
    }

    results_dir.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        results_dir / "gnn_metrics.csv", index=False)

    _plot_evaluation(all_re, scores, y_test, y_pred,
                     threshold, cm,
                     results_dir / "figures")

    return metrics


# ─────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────

def predict(X: np.ndarray, model=None, graph_data=None,
            threshold: float = None,
            save_dir: Path = MODEL_DIR) -> dict:
    """Deteksi anomali pada data baru."""
    if not TORCH_AVAILABLE:
        return {"scores": np.zeros(len(X)), "predictions": np.zeros(len(X))}

    if model is None:
        model, graph_data, threshold = load_model(save_dir)

    device     = next(model.parameters()).device
    edge_index, edge_weight, adj = graph_data

    if PYG_AVAILABLE:
        ei  = torch.LongTensor(edge_index).to(device)
        ew  = torch.FloatTensor(edge_weight).to(device)
    else:
        adj_t = torch.FloatTensor(adj).to(device)

    NF    = extract_node_features(X)
    all_re = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(NF), 64):
            batch = torch.FloatTensor(NF[i:i+64]).to(device)
            if PYG_AVAILABLE:
                for j in range(len(batch)):
                    x_hat = model(batch[j], ei, ew)
                    all_re.append(((batch[j]-x_hat)**2).mean().item())
            else:
                x_hat = model(batch, adj_t)
                all_re.extend(((batch-x_hat)**2)
                              .mean(dim=(1,2)).cpu().numpy())

    all_re  = np.array(all_re)
    re_min, re_max = all_re.min(), all_re.max()
    scores  = (all_re - re_min) / (re_max - re_min + 1e-9)
    preds   = (all_re > threshold).astype(int)

    return {"reconstruction_error": all_re,
            "scores": scores, "predictions": preds,
            "threshold": threshold}


def load_model(save_dir: Path = MODEL_DIR):
    """Load model tersimpan."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch diperlukan.")

    ckpt = torch.load(save_dir/"gnn_model_best.pt", map_location="cpu")
    hp   = ckpt["hp"]

    model = GNNAutoencoder(
        node_feat_size  = hp["node_feat_size"],
        hidden_channels = hp["hidden_channels"],
        n_layers        = hp["n_gcn_layers"],
        dropout         = hp["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    edge_index  = np.array(ckpt["edge_index"])
    edge_weight = np.array(ckpt["edge_weight"])
    adj         = np.array(ckpt["adj_matrix"])

    threshold = None
    tp = save_dir / "gnn_threshold.json"
    if tp.exists():
        with open(tp) as f:
            threshold = json.load(f)["threshold"]

    print(f"  Model loaded: val_loss={ckpt['best_val_loss']:.6f}, "
          f"threshold={threshold}")
    return model, (edge_index, edge_weight, adj), threshold


# ─────────────────────────────────────────────────────────
# HELPER PLOTS
# ─────────────────────────────────────────────────────────

def _plot_graph(corr, adj, threshold, save_path):
    """Visualisasi struktur graph sensor."""
    import networkx as nx

    G = nx.Graph()
    labels = {i: CHANNEL_ALIAS.get(MAIN_CHANNELS[i], f"S{i}")
              for i in range(N_NODES)}

    for i in range(N_NODES):
        G.add_node(i)
    for i in range(N_NODES):
        for j in range(i+1, N_NODES):
            if abs(corr[i, j]) >= threshold:
                G.add_edge(i, j, weight=abs(corr[i, j]))

    # Warna node berdasarkan tipe sensor
    colors = []
    for ch in MAIN_CHANNELS:
        alias = CHANNEL_ALIAS.get(ch, "")
        if alias.startswith("AC"): colors.append("#85B7EB")
        elif alias.startswith("CA"): colors.append("#5DCAA5")
        else: colors.append("#D3D1C7")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graph structure
    pos = nx.spring_layout(G, seed=42, k=2)
    edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]
    nx.draw_networkx(G, pos, ax=axes[0],
                     labels=labels,
                     node_color=colors, node_size=600,
                     font_size=7, font_weight="bold",
                     width=edge_weights, alpha=0.85,
                     edge_color="#888780")
    axes[0].set_title(
        f"Graph Sensor (threshold={threshold})\n"
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
        f"Biru=Accelerometer, Hijau=Cable"
    )
    axes[0].axis("off")

    # Correlation heatmap
    short = [CHANNEL_ALIAS.get(c, c.replace("FB_","")) for c in MAIN_CHANNELS]
    import seaborn as sns
    mask = np.eye(len(corr), dtype=bool)
    sns.heatmap(corr, ax=axes[1], cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True,
                xticklabels=short, yticklabels=short,
                linewidths=0.2, annot=False, mask=mask)
    axes[1].set_title("Correlation Matrix\n(input untuk edge weights)")
    axes[1].tick_params(labelsize=7)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.suptitle("GNN — Struktur Graph Sensor", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path.name}")


def _plot_loss(history, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train", color="#7F77DD", lw=1.8)
    ax.plot(history["val_loss"],   label="Val",   color="#D85A30", lw=1.8)
    best = history["val_loss"].index(min(history["val_loss"]))
    ax.axvline(best, color="#BA7517", lw=1.2, linestyle="--",
               label=f"Best epoch={best+1}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("GNN Autoencoder — Loss Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_re_distribution(all_re, y_true, threshold, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(all_re[y_true==0], bins=60, alpha=0.75,
            color="#7F77DD", label="Normal")
    if y_true.sum():
        ax.hist(all_re[y_true==1], bins=40, alpha=0.75,
                color="#E24B4A", label="Abnormal")
    ax.axvline(threshold, color="#BA7517", lw=2, linestyle="--",
               label=f"Threshold={threshold:.5f}")
    ax.set_xlabel("Reconstruction Error"); ax.set_ylabel("Count")
    ax.set_title("GNN — Distribusi RE")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path.name}")


def _plot_evaluation(all_re, scores, y_test, y_pred,
                     threshold, cm, save_dir):
    from sklearn.metrics import roc_curve as _roc, roc_auc_score
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    im = axes[0].imshow(cm, cmap="Purples")
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(["Normal","Abnormal"])
    axes[0].set_yticklabels(["Normal","Abnormal"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix — GNN")
    for i in range(2):
        for j in range(2):
            axes[0].text(j,i,f"{cm[i,j]:,}", ha="center",va="center",
                         fontsize=14,fontweight="bold",
                         color="white" if cm[i,j]>cm.max()//2 else "black")
    plt.colorbar(im, ax=axes[0])

    if y_test.sum() > 0:
        fpr,tpr,_ = _roc(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        axes[1].plot(fpr,tpr, color="#7F77DD", lw=2,
                     label=f"AUC={auc:.4f}")
        axes[1].plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title("ROC Curve — GNN")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("GNN Autoencoder — Evaluasi Data Test", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir/"gnn_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Timeline
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    idx = np.arange(len(all_re))
    axes[0].plot(idx, all_re, color="#888780", lw=0.6, alpha=0.8)
    axes[0].axhline(threshold, color="#BA7517", lw=1.5, linestyle="--",
                    label=f"Threshold={threshold:.5f}")
    if y_test.sum():
        abn = np.where(y_test==1)[0]
        axes[0].scatter(abn, all_re[abn], color="#E24B4A",
                        s=6, zorder=5, label="Actual abnormal")
    fp = np.where((y_pred==1)&(y_test==0))[0]
    if len(fp):
        axes[0].scatter(fp, all_re[fp], color="#FAC775",
                        s=4, zorder=4, label="False positive")
    axes[0].set_ylabel("Reconstruction Error")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    axes[1].fill_between(idx, scores, alpha=0.6, color="#7F77DD")
    axes[1].axhline(0.5, color="#BA7517", lw=1, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Anomaly Score (0–1)")
    axes[1].set_xlabel("Window index")
    axes[1].grid(True, alpha=0.2)
    fig.suptitle("GNN — Anomaly Detection Timeline", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir/"gnn_anomaly_timeline.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plots disimpan di: {save_dir}/")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def run_phase3c(retrain: bool = False) -> dict:
    """Jalankan Phase 3C lengkap."""
    print(f"\n{'='*65}")
    print(f"  PHASE 3C — GNN AUTOENCODER")
    print(f"{'='*65}")

    model_path = MODEL_DIR / "gnn_model_best.pt"

    if not model_path.exists() or retrain:
        model, graph_data = train(HP, MODEL_DIR, RESULTS_DIR)
        if model is None:
            return {}
        threshold = calibrate_threshold(
            model, graph_data, HP, MODEL_DIR, RESULTS_DIR)
    else:
        print(f"  Model sudah ada, load dari: {model_path}")
        model, graph_data, threshold = load_model(MODEL_DIR)

    metrics = evaluate(model, graph_data, threshold,
                       HP, MODEL_DIR, RESULTS_DIR)

    print(f"\n  PHASE 3C SELESAI")
    print(f"  F1={metrics.get('f1',0):.4f} | "
          f"AUC={metrics.get('auc',0):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SHMS GNN Autoencoder")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    if args.train or args.retrain:
        model, gd = train(HP, MODEL_DIR, RESULTS_DIR)
        if model:
            thr = calibrate_threshold(model, gd, HP, MODEL_DIR, RESULTS_DIR)
            evaluate(model, gd, thr, HP, MODEL_DIR, RESULTS_DIR)
    elif args.eval:
        model, gd, thr = load_model(MODEL_DIR)
        evaluate(model, gd, thr, HP, MODEL_DIR, RESULTS_DIR)
    else:
        run_phase3c()


if __name__ == "__main__":
    main()