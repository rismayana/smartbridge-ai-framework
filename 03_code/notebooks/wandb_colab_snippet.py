"""
W&B (Weights & Biases) — Snippet untuk Google Colab
Copy-paste cell-cell ini ke notebook SHMS_Phase3A (LSTM) dan SHMS_Phase3C (GNN).

PETUNJUK:
  1. Jalankan Cell 0 sekali di awal notebook
  2. Sesuaikan config di Cell 1 dengan HP model yang digunakan
  3. Tambahkan wandb.log() di dalam training loop (Cell 2)
  4. Jalankan Cell 3 setelah training & evaluasi selesai
"""

# ══════════════════════════════════════════════════════════════════
# CELL 0 — Install & Login (jalankan sekali di awal notebook)
# ══════════════════════════════════════════════════════════════════
"""
!pip install wandb -q

import wandb

# Dapatkan API key di: https://wandb.ai/settings
# Cara aman: simpan sebagai Colab Secret (kunci di sidebar kiri)
from google.colab import userdata
WANDB_API_KEY = userdata.get("WANDB_API_KEY")   # nama secret: WANDB_API_KEY

wandb.login(key=WANDB_API_KEY)
print("W&B login berhasil.")
"""


# ══════════════════════════════════════════════════════════════════
# CELL 1 — Inisialisasi Run (sebelum training)
# ══════════════════════════════════════════════════════════════════
"""
# ── Untuk LSTM Autoencoder ───────────────────────────────────────
run = wandb.init(
    project = "shms-anomaly-detection",
    entity  = "rismayana",
    name    = "lstm_autoencoder",
    tags    = ["phase3a", "lstm", "gpu", "pytorch", "colab"],
    notes   = "LSTM Autoencoder — deteksi anomali berbasis pola temporal",
    config  = {
        "model"         : "LSTM_Autoencoder",
        "window_size"   : 1000,
        "n_channels"    : 17,
        "hidden_dim"    : 64,
        "n_layers"      : 2,
        "dropout"       : 0.2,
        "learning_rate" : 1e-3,
        "epochs"        : 50,
        "batch_size"    : 256,
        "optimizer"     : "Adam",
        "threshold_pct" : 95,
        "device"        : "cuda",
    },
    reinit = True,
)

# ── Untuk GNN Autoencoder — ganti config di atas dengan: ─────────
# run = wandb.init(
#     project = "shms-anomaly-detection",
#     entity  = "rismayana",
#     name    = "gnn_autoencoder",
#     tags    = ["phase3c", "gnn", "gpu", "pytorch", "colab"],
#     notes   = "GNN Autoencoder — deteksi anomali berbasis struktur sensor graph",
#     config  = {
#         "model"         : "GNN_Autoencoder",
#         "window_size"   : 1000,
#         "n_channels"    : 17,
#         "hidden_dim"    : 64,
#         "n_layers"      : 3,
#         "learning_rate" : 1e-3,
#         "epochs"        : 50,
#         "batch_size"    : 64,
#         "optimizer"     : "Adam",
#         "threshold_pct" : 95,
#         "device"        : "cuda",
#     },
#     reinit = True,
# )

print(f"W&B run dimulai: {run.name}")
print(f"Dashboard: {run.url}")
"""


# ══════════════════════════════════════════════════════════════════
# CELL 2 — Log metrics per epoch (tambahkan di dalam training loop)
# ══════════════════════════════════════════════════════════════════
"""
# Contoh penggunaan di dalam training loop:
#
# best_val_loss = float("inf")
# for epoch in range(config["epochs"]):
#     train_loss = ...   # hasil satu epoch training
#     val_loss   = ...   # hasil validasi
#
#     wandb.log({
#         "epoch"      : epoch,
#         "train_loss" : train_loss,
#         "val_loss"   : val_loss,
#     })
#
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         wandb.run.summary["best_val_loss"] = best_val_loss
#         wandb.run.summary["best_epoch"]    = epoch
"""


# ══════════════════════════════════════════════════════════════════
# CELL 3 — Log hasil akhir & akhiri run
# ══════════════════════════════════════════════════════════════════
"""
import os

# Log metrics evaluasi akhir
wandb.log({
    "precision"       : precision,    # float, hasil evaluate()
    "recall"          : recall,
    "f1"              : f1,
    "auc"             : auc,
    "threshold"       : threshold,
    "n_test_windows"  : n_test,
    "n_test_abnormal" : n_abnormal,
})

# Confusion matrix sebagai W&B Table
cm_table = wandb.Table(
    columns = ["", "Pred Normal", "Pred Abnormal"],
    data    = [
        ["Actual Normal",   tn, fp],
        ["Actual Abnormal", fn, tp],
    ]
)
wandb.log({"confusion_matrix": cm_table})

# Log plot evaluasi sebagai W&B Image
RESULTS_DIR = "/content/drive/MyDrive/Penelitian/shms-ai-anomaly-detection/05_results/figures"
for fig_name in ["lstm_evaluation.png", "lstm_anomaly_timeline.png",
                 "lstm_score_distribution.png"]:
    fig_path = os.path.join(RESULTS_DIR, fig_name)
    if os.path.exists(fig_path):
        wandb.log({fig_name.replace(".png", ""): wandb.Image(fig_path)})

# Log model sebagai W&B Artifact
MODEL_PATH = "/content/drive/MyDrive/Penelitian/shms-ai-anomaly-detection/04_models/lstm_best.pt"
if os.path.exists(MODEL_PATH):
    artifact = wandb.Artifact(
        name        = "lstm_autoencoder_model",   # atau "gnn_autoencoder_model"
        type        = "model",
        description = f"LSTM Autoencoder F1={f1:.4f}",
        metadata    = {"f1": f1, "auc": auc, "precision": precision, "recall": recall},
    )
    artifact.add_file(MODEL_PATH)
    wandb.log_artifact(artifact)

wandb.finish()
print(f"W&B run selesai. Dashboard: https://wandb.ai/rismayana/shms-anomaly-detection")
"""
