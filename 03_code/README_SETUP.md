# SHMS AI Anomaly Detection — Setup Guide

## Konfigurasi HD Portable

Edit `shms_config.py`, cari bagian **HD PORTABLE SUPPORT**:

```python
# Windows (contoh HD di drive E)
RAW_DATA_OVERRIDE = Path("E:/SHMS_RAW_DATA")
ABN_DATA_OVERRIDE = Path("E:/SHMS_RAW_DATA/abnormal")

# Linux (contoh HD di /media)
RAW_DATA_OVERRIDE = Path("/media/aris/SHMS_PORTABLE/raw")

# Jika data di dalam folder proyek (default)
RAW_DATA_OVERRIDE = None
```

## Struktur Folder yang Dibutuhkan di HD Portable

```
HD_PORTABLE/
├── raw/              ← file ALL_*.txt (data 10 menit)
└── abnormal/         ← file Tdata__ALL_TR_*.txt (event data)
```

## Urutan Eksekusi

| Langkah | Perintah | Keterangan |
|---|---|---|
| 1 | `python run_pipeline.py --phase batch` | Proses data raw → .npy (butuh HD) |
| 2 | Colab: Phase3A_LSTM.ipynb | LSTM training (GPU) |
| 3 | `python run_pipeline.py --phase 3b` | Isolation Forest (CPU, ~3 menit) |
| 4 | Colab: Phase3C_GNN.ipynb | GNN training (GPU) |
| 5 | `python run_pipeline.py --phase 4 5` | Ensemble + Evaluasi |

## Catatan Penting

- **Phase 1, 2, batch** → butuh data raw dari HD portable
- **Phase 3-5** → hanya butuh file `.npy` di `02_data/processed/`
- **Phase 3A & 3C** → wajib GPU, gunakan Google Colab
- **Phase 3B & 4 & 5** → bisa di CPU biasa (laptop/PC)

## Install Dependencies

```bash
# Dependencies utama (semua phase)
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib

# Untuk Phase 3A (LSTM) dan 3C (GNN) — install di Colab
pip install torch torchvision
pip install torch_geometric

# Untuk Phase 1 (PostgreSQL, opsional)
pip install sqlalchemy psycopg2-binary
```
