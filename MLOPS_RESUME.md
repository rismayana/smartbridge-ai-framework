# Resume Implementasi MLOps — SHMS Anomaly Detection

**Project:** Deteksi Anomali Jembatan Finger Bridge, Batam  
**Tanggal:** 29 Juni 2026  
**Branch aktif:** `feature/mlops-mlflow`, `feature/mlops-wandb`  
**Repository:** https://github.com/rismayana/smartbridge-ai-framework

---

## 1. Konteks Project

Project penelitian **Structural Health Monitoring System (SHMS)** untuk deteksi anomali pada jembatan Finger Bridge, Batam. Pipeline terdiri dari 5 phase:

| Phase | Deskripsi | Lokasi Eksekusi |
|-------|-----------|----------------|
| Phase 1 | Data Loading & EDA | Lokal |
| Phase 2 | Preprocessing | Lokal |
| Phase 3A | LSTM Autoencoder | Google Colab (GPU) |
| Phase 3B | Isolation Forest | Lokal (CPU) |
| Phase 3C | GNN Autoencoder | Google Colab (GPU) |
| Phase 4 | Ensemble Fusion | Lokal |
| Phase 5 | Evaluasi & Ablation Study | Lokal |

**Constraint utama:** LSTM dan GNN tidak bisa dijalankan lokal karena keterbatasan resource GPU. Hanya Isolation Forest yang bisa dijalankan di laptop.

---

## 2. Keputusan Arsitektur

### Setup Hybrid (Colab + Lokal)

Karena training dilakukan di dua tempat berbeda, dibutuhkan tracking server yang dapat diakses dari keduanya tanpa setup server sendiri.

### Dua Pendekatan — Dua Branch Terpisah

Dibuat dua branch untuk perbandingan implementasi:

| Aspek | `feature/mlops-mlflow` | `feature/mlops-wandb` |
|-------|------------------------|----------------------|
| Experiment Tracking | MLflow via DagsHub | W&B cloud |
| DVC Remote | DagsHub storage | Google Drive |
| Akun tambahan | DagsHub | W&B |
| Confusion matrix | CSV artifact | W&B Table (interaktif) |
| Plot evaluasi | Artifact file | W&B Image (preview) |
| Mode offline | Tidak ada | Ada (`wandb offline`) |
| Model Registry | Ada (Priority 5) | Tidak ada |

### Struktur Branch

```
main
├── feature/mlops-mlflow   (Priority 1–5 selesai)
└── feature/mlops-wandb    (Priority 1–4 selesai)
```

---

## 3. Implementasi per Prioritas

### Priority 1 — Experiment Tracking

#### Branch MLflow (`feature/mlops-mlflow`)

**File baru: `03_code/mlflow_config.py`**
- Helper setup koneksi ke DagsHub sebagai MLflow tracking server
- Auto-load `DAGSHUB_TOKEN` dari file `.env` atau environment variable
- Fallback ke tracking lokal (`mlruns/`) jika token tidak tersedia
- Membaca `experiment_name` dari `config.yaml`

**Modifikasi: `03_code/shms_phase3b_iforest.py`**
- Ditambahkan fungsi `_log_to_mlflow()` yang dipanggil otomatis setelah `run_phase3b()`
- Log hyperparameter (n_estimators, contamination, threshold_pct, dll.)
- Log metrics evaluasi (F1, AUC, Precision, Recall, confusion matrix)
- Log plot evaluasi sebagai artifact
- Log model sklearn ke MLflow

**File baru: `03_code/notebooks/mlflow_colab_snippet.py`**
- 4 cell siap copy-paste ke notebook Colab (LSTM & GNN)
- Cell 0: Install & setup koneksi DagsHub
- Cell 1: Init run + log hyperparameter
- Cell 2: Log metrics per epoch (di dalam training loop)
- Cell 3: Log hasil akhir + artifacts
- Cell 4: Registrasi model ke Model Registry

#### Branch W&B (`feature/mlops-wandb`)

**File baru: `03_code/wandb_config.py`**
- Helper setup koneksi ke W&B cloud
- Auto-load `WANDB_API_KEY` dari file `.env`
- Mode offline otomatis jika tidak ada koneksi (bisa sync nanti)
- Membaca `wandb_project` dan `wandb_entity` dari `config.yaml`

**Modifikasi: `03_code/shms_phase3b_iforest.py`**
- Ditambahkan fungsi `_log_to_wandb()` yang dipanggil otomatis setelah `run_phase3b()`
- Log config, metrics, confusion matrix sebagai W&B Table interaktif
- Log plot evaluasi sebagai W&B Image (preview langsung di dashboard)
- Log model sebagai W&B Artifact (termasuk scaler dan HP JSON)

**File baru: `03_code/notebooks/wandb_colab_snippet.py`**
- 4 cell siap copy-paste ke notebook Colab (LSTM & GNN)
- Cell 0: Install & login W&B (via Colab Secrets)
- Cell 1: Init run + config (template LSTM & GNN)
- Cell 2: Log metrics per epoch
- Cell 3: Log hasil akhir + artifact model

---

### Priority 2 — Data Versioning (DVC)

Dilakukan di **kedua branch** dengan perbedaan remote storage:

| File | Keterangan |
|------|-----------|
| `.dvc/config` | Konfigurasi DVC remote (DagsHub atau Google Drive) |
| `.dvcignore` | Exclude cache, `__pycache__`, checkpoint |
| `.env.example` | Template credentials (salin ke `.env`, jangan di-commit) |
| `.gitignore` | Ditambahkan `mlruns/`, `wandb/` |

> **Catatan teknis:** `dvc init` gagal di Google Drive karena semaphore timeout pada Windows network filesystem. File konfigurasi DVC dibuat manual dengan hasil yang setara.

**Cara pakai DVC:**
```bash
# Pertama kali — tambahkan data ke DVC tracking
dvc add 02_data/processed/
dvc add 04_models/

# Upload ke remote
dvc push

# Download di tempat lain (Colab, PC lain)
dvc pull
```

---

### Priority 3 — Config Terpusat

Dilakukan di **kedua branch** dengan perbedaan di section `mlops`.

**File baru: `config.yaml`** (root project)

Satu file sebagai sumber tunggal semua hyperparameter:

```yaml
project:
  name: shms-anomaly-detection
  bridge: Finger Bridge, Batam

data:
  window_size: 1000       # 10 detik × 100 Hz
  window_step: 500        # overlap 50%
  n_channels: 17          # 9 accel + 8 cable

models:
  iforest:
    n_estimators: 200
    contamination: 0.05
    threshold_pct: 95
    # ... dll

  lstm:
    hidden_size: 64
    n_epochs: 50
    learning_rate: 0.001
    # ... dll

  gnn:
    hidden_channels: 32
    corr_threshold: 0.5
    # ... dll

  ensemble:
    weight_method: f1

mlops:
  tracking: mlflow          # atau "wandb" di branch W&B
  experiment_name: shms-anomaly-detection
  # ... credentials config
```

**File baru: `03_code/config_loader.py`**
- Load dan cache `config.yaml` menggunakan `lru_cache` (tidak baca ulang tiap call)
- Helper functions: `get_hp("iforest")`, `get_data_config()`, `get_mlops_config()`
- Fallback graceful jika `config.yaml` atau `pyyaml` tidak ada
- Mendukung dua lokasi: root project atau folder `03_code/`

**Modifikasi: `03_code/shms_phase3b_iforest.py`**
- HP dict tetap ada sebagai fallback default
- Saat startup, HP di-override dari `config.yaml` secara otomatis
- Backward-compatible: script tetap berjalan tanpa `config.yaml`

```python
# Cara ubah hyperparameter tanpa sentuh kode:
# Edit config.yaml → models.iforest.n_estimators: 300
# Jalankan → otomatis pakai nilai baru
python 03_code/run_pipeline.py --phase 3b
```

**Penambahan ke `requirements.txt`:**
- `pyyaml>=6.0`
- `python-dotenv>=1.0`

---

### Priority 4 — Pipeline Automation

Dilakukan di **kedua branch** dengan target yang sedikit berbeda di section MLOps.

**File baru: `Makefile`** (untuk Unix / Git Bash)

**File baru: `tasks.ps1`** (PowerShell — langsung bisa dipakai di Windows)

> **Catatan:** `make` tidak terinstall di lingkungan ini. `tasks.ps1` dibuat sebagai solusi Windows-native yang bisa digunakan langsung.

Daftar target yang tersedia:

| Target | Fungsi |
|--------|--------|
| `help` | Tampilkan semua perintah (default) |
| `install` | `pip install -r requirements.txt` |
| `check-config` | Verifikasi `config.yaml` terbaca dengan benar |
| `check-mlflow` / `check-wandb` | Cek koneksi ke tracking server |
| `data-pull` | Download data via DVC dari remote |
| `data-push` | Upload data via DVC ke remote |
| `preprocess` | Phase 1 + 2 (EDA + preprocessing) |
| `batch` | Batch processor data raw dari HD portable |
| `train` | Phase 3B — Isolation Forest |
| `retrain` | Phase 3B — paksa training ulang |
| `ensemble` | Phase 4 — Ensemble fusion |
| `evaluate` | Phase 5 — Evaluasi & ablation study |
| `pipeline` | Phase 3B + 4 + 5 sekaligus |
| `mlflow-ui` | Buka MLflow UI lokal di port 5000 |
| `wandb-sync` | Sync run offline ke wandb.ai |
| `wandb-check` | Verifikasi login W&B |

**Cara pakai:**
```powershell
# PowerShell (Windows)
.\tasks.ps1 help
.\tasks.ps1 train
.\tasks.ps1 pipeline

# Bash / Git Bash (jika make terinstall)
make help
make train
make pipeline
```

---

### Priority 5 — MLflow Model Registry *(hanya `feature/mlops-mlflow`)*

**File baru: `03_code/model_registry.py`**

Helper lengkap untuk manajemen model di MLflow Model Registry (DagsHub):

| Fungsi | Kegunaan |
|--------|----------|
| `register_model()` | Daftarkan model dari run ID ke registry |
| `register_best_iforest()` | Auto-cari run terbaik (F1 tertinggi) lalu daftarkan |
| `promote_model()` | Promosikan versi ke Staging / Production / Archived |
| `load_production_model()` | Load model Production untuk inference |
| `list_models()` | Tampilkan semua model + versi + stage |

**Alur model registry:**
```
Training selesai
      |
      v (otomatis)
   Staging          <- validasi manual, cek metrics di DagsHub
      |
      v (manual: registry-promote)
  Production        <- siap digunakan untuk inference
```

**Nama model standar di registry:**

| Model | Nama di Registry |
|-------|----------------|
| Isolation Forest | `shms-isolation-forest` |
| LSTM Autoencoder | `shms-lstm-autoencoder` |
| GNN Autoencoder | `shms-gnn-autoencoder` |
| Ensemble | `shms-ensemble` |

**Modifikasi: `03_code/shms_phase3b_iforest.py`**
- Setelah training selesai, model **otomatis terdaftar ke Staging** di registry
- Tidak perlu langkah manual tambahan setelah `run_phase3b()`

**Target baru di `tasks.ps1` dan `Makefile`:**
```powershell
.\tasks.ps1 registry-list      # lihat semua model + versi + stage
.\tasks.ps1 registry-register  # daftarkan IForest terbaik ke Staging
.\tasks.ps1 registry-promote   # promosikan ke Production
```

**CLI langsung:**
```bash
python 03_code/model_registry.py list
python 03_code/model_registry.py register --stage Staging
python 03_code/model_registry.py promote iforest 1 --stage Production
python 03_code/model_registry.py load iforest
```

---

## 4. Daftar Seluruh File yang Dibuat / Dimodifikasi

### Branch `feature/mlops-mlflow` (14 file, +1.132 baris)

| File | Status | Keterangan |
|------|--------|-----------|
| `03_code/mlflow_config.py` | Baru | Setup MLflow + DagsHub |
| `03_code/config_loader.py` | Baru | Load config.yaml |
| `03_code/model_registry.py` | Baru | MLflow Model Registry helper |
| `03_code/notebooks/mlflow_colab_snippet.py` | Baru | Snippet Colab (LSTM & GNN) |
| `03_code/shms_phase3b_iforest.py` | Modifikasi | Tambah MLflow logging + auto-register |
| `config.yaml` | Baru | Centralized hyperparameter config |
| `Makefile` | Baru | Pipeline automation (Unix/Git Bash) |
| `tasks.ps1` | Baru | Pipeline automation (PowerShell/Windows) |
| `.dvc/config` | Baru | DVC remote ke DagsHub |
| `.dvc/.gitignore` | Baru | Gitignore untuk folder DVC |
| `.dvcignore` | Baru | DVC ignore rules |
| `.env.example` | Baru | Template credentials |
| `.gitignore` | Modifikasi | Tambah `mlruns/` |
| `requirements.txt` | Modifikasi | Tambah mlflow, dvc, pyyaml, python-dotenv |

### Branch `feature/mlops-wandb` (13 file, +859 baris)

| File | Status | Keterangan |
|------|--------|-----------|
| `03_code/wandb_config.py` | Baru | Setup W&B |
| `03_code/config_loader.py` | Baru | Load config.yaml (sama) |
| `03_code/notebooks/wandb_colab_snippet.py` | Baru | Snippet Colab (LSTM & GNN) |
| `03_code/shms_phase3b_iforest.py` | Modifikasi | Tambah W&B logging |
| `config.yaml` | Baru | Centralized config (section mlops berbeda) |
| `Makefile` | Baru | Pipeline automation (target W&B) |
| `tasks.ps1` | Baru | Pipeline automation (target W&B) |
| `.dvc/config` | Baru | DVC remote ke Google Drive |
| `.dvc/.gitignore` | Baru | Gitignore untuk folder DVC |
| `.dvcignore` | Baru | DVC ignore rules |
| `.env.example` | Baru | Template WANDB_API_KEY |
| `.gitignore` | Modifikasi | Tambah `mlruns/`, `wandb/` |
| `requirements.txt` | Modifikasi | Tambah wandb, dvc, pyyaml, python-dotenv |

---

## 5. Status Akhir

| Priority | Fitur | MLflow Branch | W&B Branch |
|----------|-------|:---:|:---:|
| 1 | Experiment Tracking | MLflow via DagsHub | W&B cloud |
| 2 | Data Versioning | DVC + DagsHub | DVC + Google Drive |
| 3 | Config Terpusat | `config.yaml` | `config.yaml` |
| 4 | Pipeline Automation | `Makefile` + `tasks.ps1` | `Makefile` + `tasks.ps1` |
| 5 | Model Registry | `model_registry.py` | — |

Kedua branch sudah di-push ke GitHub:
- `origin/feature/mlops-mlflow`
- `origin/feature/mlops-wandb`

---

## 6. Langkah Selanjutnya

### Setup Awal (wajib sebelum bisa digunakan)

**Untuk branch MLflow:**
1. Daftar di [dagshub.com](https://dagshub.com) → connect repo `smartbridge-ai-framework`
2. Buat file `.env` dari template: `cp .env.example .env`
3. Isi `DAGSHUB_TOKEN` dari DagsHub Settings → Tokens
4. Test: `.\tasks.ps1 check-mlflow`

**Untuk branch W&B:**
1. Daftar di [wandb.ai](https://wandb.ai) → Settings → API Keys
2. Buat file `.env` dari template: `cp .env.example .env`
3. Isi `WANDB_API_KEY`
4. Test: `.\tasks.ps1 check-wandb`

### Integrasi ke Google Colab

```python
# Di awal notebook Colab — clone repo dan checkout branch
!git clone https://github.com/rismayana/smartbridge-ai-framework.git
%cd smartbridge-ai-framework
!git checkout feature/mlops-mlflow   # atau feature/mlops-wandb

# Copy snippet ke cell baru
# Lihat: 03_code/notebooks/mlflow_colab_snippet.py
#        03_code/notebooks/wandb_colab_snippet.py
```

### DVC — Tracking Data

```bash
# Pertama kali — tambahkan data ke DVC (lokal)
dvc add 02_data/processed/
dvc add 04_models/
git add 02_data/processed.dvc 04_models.dvc .gitignore
git commit -m "chore: track data dan model via DVC"

# Upload ke remote
.\tasks.ps1 data-push

# Di Colab — download data
.\tasks.ps1 data-pull   # atau: dvc pull
```

### Memilih Branch untuk Merge ke Main

Setelah keduanya dicoba, pilih satu dan buat Pull Request:
- Buka: `github.com/rismayana/smartbridge-ai-framework/compare`
- Base: `main` ← Compare: `feature/mlops-mlflow` (atau wandb)
- Merge setelah direview

---

*Dokumen ini dibuat otomatis dari resume implementasi sesi kerja MLOps, 29 Juni 2026.*
