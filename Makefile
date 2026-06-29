# SHMS Bridge Anomaly Detection - Pipeline Automation (W&B + Google Drive DVC)
# Cara pakai: make <target>
# Jika make belum terinstall di Windows: gunakan tasks.ps1
#   .\tasks.ps1 <target>   (PowerShell)

PYTHON  = python
CODE    = 03_code
PIPELINE= $(CODE)/run_pipeline.py
CFG     = $(CODE)/config_loader.py

.DEFAULT_GOAL := help

# ────────────────────────────────────────────────────────────────
# HELP
# ────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  SHMS Anomaly Detection - Pipeline Tasks (W&B + Google Drive DVC)"
	@echo "  =================================================================="
	@echo ""
	@echo "  SETUP"
	@echo "    install        Install semua dependencies (requirements.txt)"
	@echo "    check-config   Verifikasi config.yaml terbaca dengan benar"
	@echo "    check-wandb    Cek koneksi W&B"
	@echo ""
	@echo "  DATA (DVC)"
	@echo "    data-pull      Pull data processed dari Google Drive remote"
	@echo "    data-push      Push data processed ke Google Drive remote"
	@echo ""
	@echo "  PIPELINE"
	@echo "    preprocess     Phase 1 + 2 (EDA + preprocessing)"
	@echo "    batch          Batch processor (data raw dari HD portable)"
	@echo "    train          Phase 3B - Isolation Forest (lokal, CPU)"
	@echo "    retrain        Phase 3B - paksa training ulang"
	@echo "    ensemble       Phase 4 - Ensemble fusion"
	@echo "    evaluate       Phase 5 - Evaluasi dan ablation study"
	@echo "    pipeline       Phase 3B + 4 + 5 (pipeline lokal penuh)"
	@echo ""
	@echo "  W&B"
	@echo "    wandb-sync     Sync run offline ke wandb.ai"
	@echo "    wandb-check    Cek status login W&B"
	@echo ""
	@echo "  Untuk LSTM dan GNN: jalankan notebook di Google Colab"
	@echo "    03_code/notebooks/SHMS_Phase3A_LSTM_Autoencoder.ipynb"
	@echo "    03_code/notebooks/SHMS_Phase3C_GNN.ipynb"
	@echo ""

# ────────────────────────────────────────────────────────────────
# SETUP
# ────────────────────────────────────────────────────────────────
.PHONY: install
install:
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: check-config
check-config:
	$(PYTHON) $(CFG)

.PHONY: check-wandb
check-wandb:
	$(PYTHON) -c "import wandb; print('W&B version:', wandb.__version__); wandb.login()"

# ────────────────────────────────────────────────────────────────
# DATA - DVC
# ────────────────────────────────────────────────────────────────
.PHONY: data-pull
data-pull:
	$(PYTHON) -m dvc pull

.PHONY: data-push
data-push:
	$(PYTHON) -m dvc push

# ────────────────────────────────────────────────────────────────
# PIPELINE PHASES
# ────────────────────────────────────────────────────────────────
.PHONY: batch
batch:
	$(PYTHON) $(PIPELINE) --phase batch

.PHONY: preprocess
preprocess:
	$(PYTHON) $(PIPELINE) --phase 1 2

.PHONY: train
train:
	$(PYTHON) $(PIPELINE) --phase 3b

.PHONY: retrain
retrain:
	$(PYTHON) $(PIPELINE) --phase 3b --retrain

.PHONY: ensemble
ensemble:
	$(PYTHON) $(PIPELINE) --phase 4

.PHONY: evaluate
evaluate:
	$(PYTHON) $(PIPELINE) --phase 5

.PHONY: pipeline
pipeline:
	$(PYTHON) $(PIPELINE) --phase 3b 4 5

# ────────────────────────────────────────────────────────────────
# W&B
# ────────────────────────────────────────────────────────────────
.PHONY: wandb-sync
wandb-sync:
	$(PYTHON) -m wandb sync wandb/

.PHONY: wandb-check
wandb-check:
	$(PYTHON) -m wandb login --verify
