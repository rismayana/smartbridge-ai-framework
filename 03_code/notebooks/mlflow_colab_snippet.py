"""
MLflow + DagsHub — Snippet untuk Google Colab
Copy-paste cell-cell ini ke notebook SHMS_Phase3A (LSTM) dan SHMS_Phase3C (GNN).

PETUNJUK:
  1. Jalankan Cell 0 sekali di awal notebook
  2. Ganti nama run_name sesuai model (lstm_autoencoder / gnn_autoencoder)
  3. Tambahkan mlflow.log_metrics() di dalam training loop (Cell 2)
  4. Jalankan Cell 3 setelah training selesai
"""

# ══════════════════════════════════════════════════════════════════
# CELL 0 — Install & Setup (jalankan sekali di awal notebook)
# ══════════════════════════════════════════════════════════════════
"""
!pip install mlflow dagshub -q

import os
import mlflow

# ── Credentials DagsHub ──────────────────────────────────────────
# Dapatkan token di: https://dagshub.com/user/settings/tokens
DAGSHUB_USERNAME = "rismayana"
DAGSHUB_REPO     = "shms-ai-anomaly-detection"
DAGSHUB_TOKEN    = "GANTI_DENGAN_TOKEN_DAGSHUB_KAMU"   # ← isi di sini

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
mlflow.set_experiment("shms-anomaly-detection")
print("MLflow siap. Tracking ke DagsHub.")
"""


# ══════════════════════════════════════════════════════════════════
# CELL 1 — Mulai MLflow Run (sebelum training)
# ══════════════════════════════════════════════════════════════════
"""
# Ganti run_name sesuai model yang sedang dilatih
RUN_NAME = "lstm_autoencoder"    # atau "gnn_autoencoder"

mlflow_run = mlflow.start_run(run_name=RUN_NAME)

# Log hyperparameter — sesuaikan dengan HP yang digunakan
mlflow.log_params({
    # LSTM Autoencoder
    "model"        : "LSTM_Autoencoder",
    "window_size"  : 1000,
    "hidden_dim"   : 64,
    "n_layers"     : 2,
    "dropout"      : 0.2,
    "learning_rate": 1e-3,
    "epochs"       : 50,
    "batch_size"   : 256,
    "optimizer"    : "Adam",
    "threshold_pct": 95,
    "n_channels"   : 17,

    # Untuk GNN — ganti blok di atas dengan:
    # "model"        : "GNN_Autoencoder",
    # "hidden_dim"   : 64,
    # "n_layers"     : 3,
    # "learning_rate": 1e-3,
    # "epochs"       : 50,
    # "batch_size"   : 64,
})
print(f"MLflow run dimulai: {RUN_NAME}")
"""


# ══════════════════════════════════════════════════════════════════
# CELL 2 — Log metrics per epoch (tambahkan di dalam training loop)
# ══════════════════════════════════════════════════════════════════
"""
# Contoh penggunaan di dalam training loop:
#
# for epoch in range(num_epochs):
#     train_loss = ...  # hasil training
#     val_loss   = ...  # hasil validasi
#
#     mlflow.log_metrics({
#         "train_loss": train_loss,
#         "val_loss"  : val_loss,
#     }, step=epoch)
"""


# ══════════════════════════════════════════════════════════════════
# CELL 3 — Log hasil akhir & akhiri run (setelah training selesai)
# ══════════════════════════════════════════════════════════════════
"""
import mlflow.pytorch

# Log metrics evaluasi akhir
mlflow.log_metrics({
    "precision"       : precision,   # float, hasil evaluate()
    "recall"          : recall,
    "f1"              : f1,
    "auc"             : auc,
    "threshold"       : threshold,
    "n_test_windows"  : n_test,
    "n_test_abnormal" : n_abnormal,
})

# Log model PyTorch
mlflow.pytorch.log_model(model, RUN_NAME)

# Log file model dari GDrive (jika disimpan sebagai .pt)
MODEL_PATH = "/content/drive/MyDrive/Penelitian/shms-ai-anomaly-detection/04_models/lstm_best.pt"
if os.path.exists(MODEL_PATH):
    mlflow.log_artifact(MODEL_PATH, artifact_path="model_files")

# Log plot evaluasi jika ada
import glob
for fig_path in glob.glob("/content/drive/MyDrive/Penelitian/shms-ai-anomaly-detection/05_results/figures/lstm_*.png"):
    mlflow.log_artifact(fig_path, artifact_path="figures")

mlflow.end_run()
print(f"MLflow run selesai. Lihat hasil di: https://dagshub.com/rismayana/shms-ai-anomaly-detection")
"""
