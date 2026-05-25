"""
SHMS Bridge Anomaly Detection — Finger Bridge, Batam
Configuration — satu-satunya file yang perlu diedit saat setup

Last updated: 2026-04-01
"""

from pathlib import Path
from datetime import date

# ─────────────────────────────────────────────────────────
# PATHS — sesuaikan dengan lokasi folder di PC/laptop kamu
# ─────────────────────────────────────────────────────────

# Root folder proyek (otomatis detect dari lokasi file ini)
PROJECT_ROOT = Path(__file__).parent.parent

DATA_RAW_DIR      = PROJECT_ROOT / "02_data" / "raw"
DATA_ABNORMAL_DIR = PROJECT_ROOT / "02_data" / "abnormal"
DATA_STAT_DIR     = PROJECT_ROOT / "02_data" / "stat"
DATA_PROCESSED_DIR= PROJECT_ROOT / "02_data" / "processed"
MODEL_DIR         = PROJECT_ROOT / "04_models"
RESULTS_DIR       = PROJECT_ROOT / "05_results"

for d in [DATA_RAW_DIR, DATA_ABNORMAL_DIR, DATA_STAT_DIR,
          DATA_PROCESSED_DIR, MODEL_DIR, RESULTS_DIR,
          RESULTS_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# FORMAT FILE — dari DAQ sistem
# ─────────────────────────────────────────────────────────

# Raw data 10 menit
# Contoh: ALL_20260205203000.txt  (ekstensi .txt tapi isi CSV)
RAW_FILE_PATTERN   = "ALL_*.txt"
RAW_FILE_PREFIX    = "ALL_"
RAW_DATETIME_FMT   = "%Y%m%d%H%M%S"   # posisi setelah prefix
# Contoh parse: "ALL_20260205203000.txt" → datetime(2026,2,5,20,30,0)

# Data abnormal / event
# Contoh: Tdata__ALL_TR_20260206_101144.txt
ABNORMAL_FILE_PATTERN  = "Tdata__ALL_TR_*.txt"
ABNORMAL_FILE_PREFIX   = "Tdata__ALL_TR_"
ABNORMAL_DATETIME_FMT  = "%Y%m%d_%H%M%S"
# Contoh parse: "Tdata__ALL_TR_20260206_101144.txt" → datetime(2026,2,6,10,11,44)

# Encoding & separator (hasil diagnosis sebelumnya)
FILE_ENCODING = "latin-1"
FILE_NA_VALUES = ["NaN", " ", ""]

# ─────────────────────────────────────────────────────────
# PERIODE DATA
# ─────────────────────────────────────────────────────────

DATA_START = date(2026, 2, 6)    # tanggal mulai data tersedia
DATA_END   = date(2026, 3, 29)   # tanggal akhir data tersedia

# Split train / validation / test (berbasis waktu, bukan random)
TRAIN_END  = date(2026, 2, 28)   # training  : 6 Feb – 28 Feb (21 hari)
VAL_END    = date(2026, 3, 14)   # validation: 1 Mar – 14 Mar (14 hari)
# test                           # test      : 15 Mar – 29 Mar (15 hari)

# ─────────────────────────────────────────────────────────
# POSTGRESQL — statistik 10 menit
# ─────────────────────────────────────────────────────────

PG_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "database": "shms_db",
    "user"    : "postgres",
    "password": "your_password",
}
PG_STAT_TABLE = "sensor_statistics"

# ─────────────────────────────────────────────────────────
# SENSOR METADATA
# ─────────────────────────────────────────────────────────

SENSOR_TYPE_MAP = {
    "CA": "Cable Tensionmeter",
    "AC": "Accelerometer",
    "DS": "Strain Meter",
    "TP": "Member Thermometer",
    "EX": "Joint Meter",
    "GP": "GNSS",
    "AT": "Air Temp",
    "WG": "Wind Gauge",
    "RH": "Humidity",
}

# ─────────────────────────────────────────────────────────
# DATASET FINAL — 20 CHANNEL
# Hasil diagnosis per 29 Mar 2026
# ─────────────────────────────────────────────────────────

SELECTED_CHANNELS = {
    # == ACCELEROMETER — 9 channel (Gal = cm/s²) ==
    "AC_PY1T_BX" : "FB_AC_PY1T_01_BX",   # Pylon 1 Top — lateral
    "AC_PY1T_BY" : "FB_AC_PY1T_01_BY",   # Pylon 1 Top — longitudinal
    "AC_PY1D_BX" : "FB_AC_PY1D_01_BX",   # Pylon 1 Deck — lateral
    "AC_PY1D_BY" : "FB_AC_PY1D_01_BY",   # Pylon 1 Deck — longitudinal
    "AC_S2M_BZ"  : "FB_AC_S2M_01_BZ",    # Mid Span 2 — vertikal
    "AC_S1Q1_BX" : "FB_AC_S1Q1_01_BX",   # Span 1 Quarter — lateral
    "AC_S1Q1_BZ" : "FB_AC_S1Q1_01_BZ",   # Span 1 Quarter — vertikal
    "AC_S3Q3_BY" : "FB_AC_S3Q3_01_BY",   # Span 3 Quarter — longitudinal
    "AC_S3Q3_BZ" : "FB_AC_S3Q3_01_BZ",   # Span 3 Quarter — vertikal

    # == CABLE TENSIONMETER — 8 channel (kN) ==
    "CA_L17"     : "FB_CA_L17_EZ",        # Kabel mid-group, kiri
    "CA_R17"     : "FB_CA_R17_EZ",        # Kabel mid-group, kanan
    "CA_L22"     : "FB_CA_L22_EZ",        # Kabel panjang, kiri
    "CA_R22"     : "FB_CA_R22_EZ",        # Kabel panjang, kanan
    "CA_L46"     : "FB_CA_L46_EZ",        # Kabel Span 3, kiri
    "CA_R46"     : "FB_CA_R46_EZ",        # Kabel Span 3, kanan
    "CA_L02"     : "FB_CA_L02_EZ",        # Anchor cable Abutment 1
    "CA_L55"     : "FB_CA_L55_EZ",        # Anchor cable Abutment 2

    # == TEMPERATURE — 2 channel (°C, kontekstual) ==
    "TP_S1Q1_02" : "FB_TP_S1Q1_02",       # Suhu deck Span 1 titik 2
    "TP_S1Q1_03" : "FB_TP_S1Q1_03",       # Suhu deck Span 1 titik 3

    # == ANEMOMETER — 1 channel (m/s, kontekstual) ==
    "WG_S2M_S"   : "FB_WG_S2M_01_S",      # Wind speed Mid Span 2
}

ALL_CHANNELS     = list(SELECTED_CHANNELS.values())
CHANNEL_ALIAS    = {v: k for k, v in SELECTED_CHANNELS.items()}

ACCEL_CHANNELS   = [v for k, v in SELECTED_CHANNELS.items() if k.startswith("AC_")]
CABLE_CHANNELS   = [v for k, v in SELECTED_CHANNELS.items() if k.startswith("CA_")]
TEMP_CHANNELS    = [v for k, v in SELECTED_CHANNELS.items() if k.startswith("TP_")]
WIND_CHANNELS    = [v for k, v in SELECTED_CHANNELS.items() if k.startswith("WG_")]
CONTEXT_CHANNELS = TEMP_CHANNELS + WIND_CHANNELS
MAIN_CHANNELS    = ACCEL_CHANNELS + CABLE_CHANNELS   # input model utama

# ─────────────────────────────────────────────────────────
# PREPROCESSING PARAMETERS
# ─────────────────────────────────────────────────────────

SAMPLE_RATE  = 100        # Hz — sensor utama
WINDOW_SEC   = 10         # detik per window
WINDOW_SIZE  = SAMPLE_RATE * WINDOW_SEC   # = 1000 sampel
WINDOW_STEP  = WINDOW_SIZE // 2           # = 500 (overlap 50%)
MAX_NAN_PCT  = 0.10       # window dengan >10% NaN → dibuang

# Batch processing — proses N file sekaligus (hemat RAM)
BATCH_FILES  = 144        # 1 hari = 144 file × 80MB = ~11GB per batch

# ─────────────────────────────────────────────────────────
# CHANNEL YANG DIKELUARKAN & ALASANNYA (untuk dokumentasi)
# ─────────────────────────────────────────────────────────

EXCLUDED_CHANNELS = {
    "FB_AC_PY2T_01_BX" : "offset 1732 Gal — masalah kalibrasi hardware",
    "FB_AC_PY2D_01_BX" : "flatline std=0 — sensor tidak merespons",
    "FB_AC_PY2D_01_BY" : "flatline std=0 — sensor tidak merespons",
    "FB_AC_S2M_01_BY"  : "mean -92 Gal — tidak dapat dijelaskan",
    "FB_AC_PY1B_01_BX" : "sensor seismik — di luar scope penelitian",
    "FB_AC_PY1B_01_BY" : "sensor seismik — di luar scope penelitian",
    "FB_AC_PY1B_01_BZ" : "sensor seismik — di luar scope penelitian",
    "FB_AC_PY2B_01_BX" : "sensor seismik + ALL NaN",
    "FB_AC_PY2B_01_BY" : "sensor seismik + ALL NaN",
    "FB_AC_PY2B_01_BZ" : "sensor seismik + ALL NaN",
    "FB_CA_R02_EZ"     : "near-zero — tidak terkoneksi ke kabel",
    "FB_CA_R07_EZ"     : "near-zero — tidak terkoneksi ke kabel",
    "FB_CA_L11_EZ"     : "near-zero — tidak terkoneksi ke kabel",
    "FB_CA_L40_EZ"     : "near-zero — tidak terkoneksi ke kabel",
    "FB_CA_R55_EZ"     : "near-zero — tidak terkoneksi ke kabel",
    "FB_CA_R27_EZ"     : "outlier 3196 kN — suspek rusak",
    "FB_CA_R30_EZ"     : "negatif -4073 kN — garbage value",
    "FB_CA_L35_EZ"     : "flatline 5807 kN — stuck value",
    "FB_CA_R50_EZ"     : "negatif -739 kN — sensor rusak",
    "FB_TP_S1Q1_01"    : "nilai 863°C — sensor rusak",
    "FB_AT_S2M_01"     : "nilai -51.25 — tidak valid",
    "FB_RH_S2M_01"     : "nilai -1.25 — tidak valid",
}



# Alias untuk kompatibilitas dengan phase2_preprocessing.py
OUTPUT_DIR = DATA_PROCESSED_DIR
# ─────────────────────────────────────────────────────────
# PATH HELPER FUNCTIONS
# Digunakan oleh phase3a, phase3c, ood_validation dll.
# Override di Colab dengan menambahkan patch di akhir file
# ─────────────────────────────────────────────────────────

def get_processed_dir() -> Path:
    """Return path ke folder processed .npy files."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_PROCESSED_DIR

def get_raw_dir() -> Path:
    """Return path ke folder raw data."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_RAW_DIR

def get_abnormal_dir() -> Path:
    """Return path ke folder abnormal data."""
    DATA_ABNORMAL_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_ABNORMAL_DIR


# ══════════════════════════════════════════════════════
# ZBOOK OVERRIDE — cek beberapa kandidat path
# Urutan: F: drive → D: drive → default PROJECT_ROOT
# ══════════════════════════════════════════════════════
def _find_processed_dir() -> Path:
    candidates = [
        Path("G:\My Drive\Penelitian\shms-ai-anomaly-detection\02_data\processed"),
        # Path("D:/Mine/Penelitian/2025/Data SHMS Batam/processed"),
        # Path("D:/Mine/Penelitian/2025/shms_03_code_zbook/02_data/processed"),
        DATA_PROCESSED_DIR,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*_X.npy")):
            return c
    # Tidak ada yang punya .npy — kembalikan default (akan masuk simulasi)
    return DATA_PROCESSED_DIR

_PROC = _find_processed_dir()
DATA_PROCESSED_DIR = _PROC
OUTPUT_DIR         = _PROC

def get_processed_dir() -> Path:
    return _PROC

def get_raw_dir() -> Path:
    for p in [Path("F:/Data SHMS Batam/RawData"),
              Path("D:/Mine/Penelitian/2025/Data SHMS Batam/RawData")]:
        if p.exists():
            return p
    return DATA_RAW_DIR

def get_abnormal_dir() -> Path:
    return DATA_ABNORMAL_DIR