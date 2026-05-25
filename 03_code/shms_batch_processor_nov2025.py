"""
SHMS Bridge Anomaly Detection
shms_batch_processor_nov2025.py — Batch processor khusus data November 2025

Perbedaan dari shms_batch_processor.py asli:
  - Input dir  : E:\\Data SHMS Batam\\Data Nov 2025  (bukan DATA_RAW_DIR)
  - Output dir : <DATA_PROCESSED_DIR>\\nov2025      (folder terpisah)
  - Tidak ada split train/val/test → semua diproses sekaligus
  - Tidak ada file abnormal (DATA_ABNORMAL_DIR) → hanya raw files
  - Progress bisa dilanjutkan jika terhenti (skip hari yang sudah ada)

Cara pakai:
  # Proses semua 30 hari November 2025:
  python shms_batch_processor_nov2025.py

  # Proses 1 hari saja (untuk test):
  python shms_batch_processor_nov2025.py --date 2025-11-14

  # Cek progress tanpa proses:
  python shms_batch_processor_nov2025.py --check

  # Paksa proses ulang (ignore cache):
  python shms_batch_processor_nov2025.py --force
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from shms_config import (
    RAW_FILE_PATTERN, RAW_FILE_PREFIX, RAW_DATETIME_FMT,
    FILE_ENCODING, FILE_NA_VALUES,
    ALL_CHANNELS, MAIN_CHANNELS, CONTEXT_CHANNELS,
    WIND_CHANNELS, WINDOW_SIZE, WINDOW_STEP, MAX_NAN_PCT,
    get_processed_dir,
)


# ─────────────────────────────────────────────────────────
# KONFIGURASI KHUSUS NOV 2025
# ─────────────────────────────────────────────────────────

NOV2025_RAW_DIR = Path(r"D:\Data SHMS Batam\Data Nov 2025")
NOV2025_START   = date(2025, 11, 1)
NOV2025_END     = date(2025, 11, 30)

# Output disimpan di subfolder nov2025 di dalam processed dir
def get_nov2025_out_dir() -> Path:
    out = get_processed_dir() / "nov2025"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────
# PARSE TANGGAL DARI NAMA FILE
# ─────────────────────────────────────────────────────────

def parse_raw_datetime(filepath: Path) -> datetime | None:
    """
    Parse datetime dari nama file.
    ALL_20251114143000.txt → datetime(2025, 11, 14, 14, 30, 0)
    """
    name   = filepath.stem
    dt_str = name.replace(RAW_FILE_PREFIX, "")
    try:
        return datetime.strptime(dt_str, RAW_DATETIME_FMT)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────
# LOAD SATU FILE
# Identik dengan batch processor asli — format file sama
# ─────────────────────────────────────────────────────────

def load_one_file(filepath: Path) -> pd.DataFrame | None:
    """
    Load satu file ALL_*.txt → DataFrame dengan 20 channel terpilih.
    Format file identik dengan data 2026 (CSV, 100 Hz, ~60.000 baris).
    """
    try:
        df = pd.read_csv(
            filepath,
            encoding   = FILE_ENCODING,
            na_values  = FILE_NA_VALUES,
            low_memory = False,
        )
    except Exception as e:
        print(f"    [ERROR] Gagal baca {filepath.name}: {e}")
        return None

    # Bersihkan nama kolom (buang whitespace, karakter non-ASCII)
    df.columns = [
        c.strip().encode("ascii", "ignore").decode("ascii").strip()
        for c in df.columns
    ]

    if df.empty or len(df) < 100:
        return None

    # Ambil hanya channel yang digunakan (20 channel)
    available = [c for c in ALL_CHANNELS if c in df.columns]
    if not available:
        print(f"    [WARN] Tidak ada channel cocok di {filepath.name}")
        return None

    out = df[available].copy()

    # Konversi ke numeric
    for col in available:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Forward-fill anemometer (sensor 1 Hz → 100 Hz)
    for col in WIND_CHANNELS:
        if col in out.columns:
            out[col] = out[col].ffill().bfill()

    return out


# ─────────────────────────────────────────────────────────
# SLIDING WINDOW
# Identik dengan batch processor asli
# ─────────────────────────────────────────────────────────

def make_windows(df: pd.DataFrame) -> np.ndarray:
    """
    Potong DataFrame menjadi sliding windows.

    Return:
        X : (n_windows, WINDOW_SIZE, n_channels) float32
            n_channels = len(MAIN_CHANNELS + CONTEXT_CHANNELS) = 20
    """
    use_cols = [c for c in MAIN_CHANNELS + CONTEXT_CHANNELS
                if c in df.columns]
    data     = df[use_cols].values.astype(np.float32)
    n_rows   = len(data)

    X_list = []

    for start in range(0, n_rows - WINDOW_SIZE + 1, WINDOW_STEP):
        end    = start + WINDOW_SIZE
        window = data[start:end].copy()

        # Buang window dengan terlalu banyak NaN
        if np.isnan(window).mean() > MAX_NAN_PCT:
            continue

        # Interpolasi NaN sisa (linear)
        for j in range(window.shape[1]):
            col = window[:, j]
            if np.isnan(col).any():
                mask  = np.isnan(col)
                valid = ~mask
                if valid.sum() > 1:
                    idx = np.arange(WINDOW_SIZE)
                    window[:, j] = np.interp(idx, idx[valid], col[valid])
                else:
                    window[:, j] = 0.0

        X_list.append(window)

    if not X_list:
        return np.empty((0, WINDOW_SIZE, len(use_cols)), dtype=np.float32)

    return np.array(X_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────
# PROSES SATU HARI
# ─────────────────────────────────────────────────────────

def process_one_day(target_date: date,
                    raw_dir: Path,
                    out_dir: Path,
                    force: bool = False,
                    verbose: bool = True) -> dict:
    """
    Proses semua file raw untuk satu hari November 2025.
    Simpan sebagai:
        YYYYMMDD_X.npy   — windows array (n, 1000, 20)
        YYYYMMDD_meta.csv — metadata

    Return: dict ringkasan hasil proses
    """
    date_str  = target_date.strftime("%Y%m%d")
    x_path    = out_dir / f"{date_str}_X.npy"
    meta_path = out_dir / f"{date_str}_meta.csv"

    # Skip jika sudah diproses (kecuali --force)
    if x_path.exists() and not force:
        meta = pd.read_csv(meta_path).iloc[0].to_dict() if meta_path.exists() else {}
        if verbose:
            print(f"  [{date_str}] Skip (sudah ada: "
                  f"{meta.get('n_windows', '?')} windows)")
        return {"date": date_str, "status": "skipped", **meta}

    # Kumpulkan file untuk hari ini
    raw_files = sorted([
        f for f in raw_dir.glob(RAW_FILE_PATTERN)
        if (dt := parse_raw_datetime(f)) and dt.date() == target_date
    ])

    if verbose:
        print(f"  [{date_str}] {len(raw_files):3d} file raw", end="")

    if not raw_files:
        if verbose:
            print(f" — tidak ada file (hari libur / mati listrik?)")
        return {"date": date_str, "status": "no_data",
                "n_files": 0, "n_windows": 0}

    # Load dan gabung semua file hari ini
    dfs       = []
    n_loaded  = 0
    n_failed  = 0

    for f in raw_files:
        df = load_one_file(f)
        if df is not None:
            dfs.append(df)
            n_loaded += 1
        else:
            n_failed += 1

    if not dfs:
        if verbose:
            print(f" — semua file gagal dibaca")
        return {"date": date_str, "status": "load_failed",
                "n_files": len(raw_files), "n_windows": 0}

    df_day = pd.concat(dfs, ignore_index=True)

    # Buat windows
    X = make_windows(df_day)

    if len(X) == 0:
        if verbose:
            print(f" — tidak ada window valid")
        return {"date": date_str, "status": "no_windows",
                "n_files": n_loaded, "n_windows": 0}

    # Simpan
    np.save(x_path, X)

    meta_row = {
        "date"        : date_str,
        "status"      : "ok",
        "n_files"     : len(raw_files),
        "n_loaded"    : n_loaded,
        "n_failed"    : n_failed,
        "n_windows"   : len(X),
        "shape_X"     : str(X.shape),
        "size_mb"     : round(X.nbytes / 1024**2, 1),
    }
    pd.DataFrame([meta_row]).to_csv(meta_path, index=False)

    if verbose:
        print(f" → {len(X):,} windows | {meta_row['size_mb']} MB")

    return meta_row


# ─────────────────────────────────────────────────────────
# PROSES SEMUA HARI
# ─────────────────────────────────────────────────────────

def process_all(raw_dir: Path, out_dir: Path,
                start: date = NOV2025_START,
                end:   date = NOV2025_END,
                force: bool = False):
    """Proses semua hari November 2025."""

    print("\n" + "="*65)
    print("  SHMS BATCH PROCESSOR — NOVEMBER 2025")
    print(f"  Periode  : {start} s/d {end}")
    print(f"  Raw dir  : {raw_dir}")
    print(f"  Out dir  : {out_dir}")
    print("="*65 + "\n")

    current  = start
    all_meta = []

    while current <= end:
        meta = process_one_day(current, raw_dir, out_dir, force=force)
        all_meta.append(meta)
        current += timedelta(days=1)

    summary = pd.DataFrame(all_meta)
    summary.to_csv(out_dir / "processing_summary_nov2025.csv", index=False)

    ok = summary[summary["status"] == "ok"]

    print("\n" + "="*65)
    print("  RINGKASAN")
    print(f"  Hari berhasil   : {len(ok)} / {(end - start).days + 1}")
    print(f"  Total windows   : {ok['n_windows'].sum():,}")

    if "size_mb" in ok.columns:
        total_mb = ok["size_mb"].sum()
        unit = "GB" if total_mb > 1024 else "MB"
        val  = total_mb/1024 if total_mb > 1024 else total_mb
        print(f"  Total .npy size : {val:.1f} {unit}")

    print("\n  Per tanggal (hari yang berhasil):")
    for _, row in ok.iterrows():
        print(f"    {row['date']} : {int(row['n_windows']):,} windows "
              f"| {row['n_loaded']} file | {row['size_mb']} MB")

    # Hari yang gagal atau tidak ada data
    not_ok = summary[summary["status"] != "ok"]
    if len(not_ok):
        print(f"\n  Hari bermasalah ({len(not_ok)}):")
        for _, row in not_ok.iterrows():
            print(f"    {row['date']} : {row['status']}")
    print("="*65)
    return summary


# ─────────────────────────────────────────────────────────
# CEK PROGRESS
# ─────────────────────────────────────────────────────────

def check_progress(out_dir: Path):
    """Tampilkan status pemrosesan tanpa proses apapun."""
    summary_path = out_dir / "processing_summary_nov2025.csv"
    if not summary_path.exists():
        print("Belum ada data yang diproses. Jalankan tanpa --check dulu.")
        return

    df    = pd.read_csv(summary_path)
    ok    = df[df["status"] == "ok"]
    total = (NOV2025_END - NOV2025_START).days + 1

    print(f"\nProgress  : {len(ok)}/{total} hari "
          f"({100*len(ok)/total:.0f}%)")
    print(f"Windows   : {ok['n_windows'].sum():,} total")
    if "size_mb" in ok.columns:
        print(f"Disk used : {ok['size_mb'].sum():.0f} MB")

    print("\nStatus per tanggal:")
    for _, row in df.iterrows():
        status_icon = "✅" if row["status"] == "ok" else "⚠️ "
        n_win = int(row["n_windows"]) if row["status"] == "ok" else 0
        print(f"  {status_icon} {row['date']} : {row['status']:12s} "
              f"({n_win:,} windows)")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch processor khusus data November 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # Proses semua 30 hari:
  python shms_batch_processor_nov2025.py

  # Proses 1 hari saja (untuk test cepat):
  python shms_batch_processor_nov2025.py --date 2025-11-14

  # Cek progress tanpa proses:
  python shms_batch_processor_nov2025.py --check

  # Paksa proses ulang hari yang sudah ada:
  python shms_batch_processor_nov2025.py --force
        """
    )
    parser.add_argument("--date",  type=str, default=None,
                        help="Proses 1 hari saja. Format: YYYY-MM-DD")
    parser.add_argument("--check", action="store_true",
                        help="Cek progress saja, tidak proses")
    parser.add_argument("--force", action="store_true",
                        help="Proses ulang meski sudah ada")
    parser.add_argument("--raw-dir", type=str,
                        default=str(NOV2025_RAW_DIR),
                        help=f"Folder raw data (default: {NOV2025_RAW_DIR})")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = get_nov2025_out_dir()

    if args.check:
        check_progress(out_dir)
        return

    if args.date:
        target = date.fromisoformat(args.date)
        process_one_day(target, raw_dir, out_dir, force=args.force)
        return

    process_all(raw_dir, out_dir, force=args.force)


if __name__ == "__main__":
    main()
