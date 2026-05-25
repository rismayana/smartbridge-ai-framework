"""
SHMS Bridge Anomaly Detection
shms_batch_processor.py — Batch processor: raw .txt → processed .npy

Optimasi v2 (dari profiling Task Manager: CPU 12%, RAM 56% stabil, Disk USB 95%):

  Masalah lama:
    - Baca 90 kolom → buang 70 → disk USB kerja 4.5x sia-sia
    - Sequential: baca file 1, selesai, baca file 2, selesai, ...
    - list.append() di windowing → banyak realokasi kecil

  Solusi baru:
    1. usecols saat read_csv  → baca 20 kolom saja, disk transfer turun 78%
    2. dtype=float32 langsung → hemat 50% RAM vs float64
    3. Pipeline I/O + CPU     → ThreadPoolExecutor: sambil windowing file i,
                                 thread lain sudah baca file i+1 dari disk
    4. Pre-allocate numpy     → tidak append ke list, tidak realokasi
    5. gc.collect() per batch → bebaskan RAM segera setelah selesai

Cara pakai:
  python shms_batch_processor.py                    # proses semua hari
  python shms_batch_processor.py --date 2026-02-06  # proses 1 hari saja
  python shms_batch_processor.py --check            # cek progress saja
  python shms_batch_processor.py --workers 3        # atur jumlah thread
"""

import argparse
import gc
import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from shms_config import (
    DATA_RAW_DIR, DATA_ABNORMAL_DIR, DATA_PROCESSED_DIR,
    get_raw_dir, get_abnormal_dir,
    RAW_FILE_PATTERN, RAW_FILE_PREFIX, RAW_DATETIME_FMT,
    ABNORMAL_FILE_PATTERN, ABNORMAL_DATETIME_FMT, ABNORMAL_FILE_PREFIX,
    SENSOR_SPECIFIC_PREFIX,
    FILE_ENCODING, FILE_NA_VALUES,
    ALL_CHANNELS, MAIN_CHANNELS, CONTEXT_CHANNELS, WIND_CHANNELS,
    WINDOW_SIZE, WINDOW_STEP, MAX_NAN_PCT,
    DATA_START, DATA_END, TRAIN_END, VAL_END,
)

# ─────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────

# Jumlah thread untuk baca file paralel.
# USB HD = I/O bound → 2-3 thread sudah optimal karena head HD tidak bisa
# baca banyak posisi sekaligus. Naikkan hanya jika SSD eksternal.
DEFAULT_WORKERS = 3

# Kolom yang perlu dibaca (set untuk lookup O(1))
_COLS_NEEDED = set(ALL_CHANNELS)

# Indeks channel utama dalam ALL_CHANNELS
_MAIN_IDX = [ALL_CHANNELS.index(c) for c in MAIN_CHANNELS if c in ALL_CHANNELS]
_CTX_IDX  = [ALL_CHANNELS.index(c) for c in CONTEXT_CHANNELS if c in ALL_CHANNELS]
_USE_IDX  = _MAIN_IDX + _CTX_IDX   # indeks yang masuk ke model


# ─────────────────────────────────────────────────────────
# PARSE DATETIME
# ─────────────────────────────────────────────────────────

def parse_raw_datetime(filepath: Path) -> datetime | None:
    name   = filepath.stem
    dt_str = name.replace(RAW_FILE_PREFIX, "")
    try:
        return datetime.strptime(dt_str, RAW_DATETIME_FMT)
    except ValueError:
        return None


def parse_abnormal_datetime(filepath: Path) -> datetime | None:
    """
    Hanya terima Tdata__ALL_TR_*.txt (kecil 'd').
    TData__KODE_SENSOR_* (kapital 'D') ditolak — duplikat dari ALL_TR.
    """
    name = filepath.stem
    if name.startswith(SENSOR_SPECIFIC_PREFIX) and "ALL_TR" not in name:
        return None
    if not name.startswith(ABNORMAL_FILE_PREFIX):
        return None
    dt_str = name.replace(ABNORMAL_FILE_PREFIX, "")
    try:
        return datetime.strptime(dt_str, ABNORMAL_DATETIME_FMT)
    except ValueError:
        return None


def get_split(file_date: date) -> str:
    if file_date <= TRAIN_END:
        return "train"
    elif file_date <= VAL_END:
        return "val"
    else:
        return "test"


# ─────────────────────────────────────────────────────────
# LOAD SATU FILE — OPTIMASI usecols + float32
# ─────────────────────────────────────────────────────────

def load_one_file(filepath: Path, is_abnormal: bool = False) -> np.ndarray | None:
    """
    Baca satu file CSV → numpy array float32 (20 channel + 1 flag).

    Sebelum: read 90 kolom (float64) → filter → 41 MB/file dari disk
    Sesudah: read 20 kolom (float32) → langsung  4.6 MB/file dari disk
    Penghematan disk read: ~89%

    Return shape: (n_rows, 21) — 20 channel + kolom flag is_abnormal
    Return None  : file kosong / corrupt / tidak ada channel yang cocok
    """
    try:
        # Baca header saja dulu (0 baris) untuk tahu kolom tersedia
        hdr = pd.read_csv(
            filepath, encoding=FILE_ENCODING,
            nrows=0, low_memory=False,
        )
        hdr.columns = [
            c.strip().encode("ascii", "ignore").decode("ascii").strip()
            for c in hdr.columns
        ]

        # Pilih kolom yang ada dan diperlukan
        usecols = [c for c in ALL_CHANNELS if c in hdr.columns]
        if not usecols:
            return None

        # Baca hanya kolom yang perlu, langsung float32
        df = pd.read_csv(
            filepath,
            encoding  = FILE_ENCODING,
            na_values = FILE_NA_VALUES,
            usecols   = usecols,
            dtype     = {c: "float32" for c in usecols},
            low_memory= False,
        )

        if df.empty or len(df) < WINDOW_SIZE // 4:
            return None

        # Forward-fill anemometer (1 Hz → ikuti 100 Hz lainnya)
        for col in WIND_CHANNELS:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Susun kolom sesuai urutan ALL_CHANNELS, pad NaN untuk yang tidak ada
        out = np.full((len(df), len(ALL_CHANNELS)), np.nan, dtype=np.float32)
        for i, col in enumerate(ALL_CHANNELS):
            if col in df.columns:
                out[:, i] = df[col].values

        # Tambah kolom flag is_abnormal (kolom ke-20, index 20)
        flag = np.full((len(out), 1),
                       1.0 if is_abnormal else 0.0, dtype=np.float32)
        return np.concatenate([out, flag], axis=1)  # shape (n_rows, 21)

    except Exception as e:
        print(f"    [ERROR] {filepath.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────
# SLIDING WINDOW — PRE-ALLOC + VECTORIZED NaN CHECK
# ─────────────────────────────────────────────────────────

def make_windows(arr: np.ndarray,
                 channel_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Buat sliding windows dari array (n_rows, 21).

    Sebelum: X_list.append() → realokasi tiap window
    Sesudah: pre-allocate X_out → isi langsung ke slot

    Args:
        arr         : (n_rows, 21) output load_one_file atau concat
        channel_idx : indeks channel yang masuk ke model

    Return:
        X : (n_windows, WINDOW_SIZE, n_channels) float32
        y : (n_windows,) int8
    """
    n_rows    = len(arr)
    flag_col  = arr.shape[1] - 1
    sensor    = arr[:, channel_idx]   # view, bukan copy
    labels    = arr[:, flag_col]

    n_max = max(0, (n_rows - WINDOW_SIZE) // WINDOW_STEP + 1)
    if n_max == 0:
        n_ch = len(channel_idx)
        return (np.empty((0, WINDOW_SIZE, n_ch), dtype=np.float32),
                np.empty((0,), dtype=np.int8))

    n_ch  = len(channel_idx)
    X_out = np.empty((n_max, WINDOW_SIZE, n_ch), dtype=np.float32)
    y_out = np.empty(n_max, dtype=np.int8)
    count = 0

    for start in range(0, n_rows - WINDOW_SIZE + 1, WINDOW_STEP):
        end    = start + WINDOW_SIZE
        window = sensor[start:end].copy()

        # Vectorized NaN check
        nan_ratio = np.isnan(window).mean()
        if nan_ratio > MAX_NAN_PCT:
            continue

        # Interpolasi NaN (hanya kalau perlu)
        if nan_ratio > 0:
            for j in range(n_ch):
                col = window[:, j]
                mask = np.isnan(col)
                if mask.any():
                    valid = ~mask
                    if valid.sum() > 1:
                        idx = np.arange(WINDOW_SIZE)
                        window[:, j] = np.interp(idx, idx[valid], col[valid])
                    else:
                        window[:, j] = 0.0

        X_out[count] = window
        y_out[count] = 1 if labels[start:end].mean() >= 0.5 else 0
        count += 1

    return X_out[:count], y_out[:count]


# ─────────────────────────────────────────────────────────
# PROSES SATU HARI — PIPELINE I/O + CPU
# ─────────────────────────────────────────────────────────

def process_one_day(target_date: date, raw_dir: Path,
                    abnormal_dir: Path, out_dir: Path,
                    verbose: bool = True,
                    n_workers: int = DEFAULT_WORKERS) -> dict:
    """
    Proses semua file raw + abnormal satu hari.

    Pipeline baru:
      Thread pool membaca file dari HD secara paralel.
      Hasil setiap file langsung di-windowing begitu selesai dibaca,
      tidak ditampung semua dulu di RAM lalu di-concat besar-besaran.

    Output: YYYYMMDD_X.npy, YYYYMMDD_y.npy, YYYYMMDD_meta.csv
    """
    date_str  = target_date.strftime("%Y%m%d")
    x_path    = out_dir / f"{date_str}_X.npy"
    y_path    = out_dir / f"{date_str}_y.npy"
    meta_path = out_dir / f"{date_str}_meta.csv"

    # Skip jika sudah ada
    if x_path.exists() and y_path.exists():
        meta = pd.read_csv(meta_path).iloc[0].to_dict() if meta_path.exists() else {}
        if verbose:
            print(f"  [{date_str}] Skip — sudah diproses "
                  f"({meta.get('n_windows','?')} windows)")
        return {"date": date_str, "status": "skipped", **meta}

    # ── Kumpulkan file ────────────────────────────────────
    raw_files = sorted([
        f for f in raw_dir.glob(RAW_FILE_PATTERN)
        if (dt := parse_raw_datetime(f)) and dt.date() == target_date
    ])

    # ── Filter ALL_TR berdasarkan sensor trigger ─────────
    # Masalah: sensor rusak (mis. FB_CA_R27) bisa trigger 4000+ event/hari
    # → ribuan ALL_TR diambil → 74% window berlabel anomali (tidak masuk akal)
    # → F1/Recall hancur karena label noise massal
    #
    # Solusi: hanya ambil ALL_TR jika sensor trigger-nya ada di SELECTED channels
    # Caranya: scan TData__KODE_SENSOR_* → cek kode → filter timestamp ALL_TR

    valid_timestamps, skipped_by_sensor = get_valid_abn_timestamps(
        abnormal_dir, target_date
    )

    # Kumpulkan ALL_TR yang timestamp-nya ada di valid set
    abn_files = []
    for f in sorted(abnormal_dir.glob("Tdata__ALL_TR_*.txt")):
        dt = parse_abnormal_datetime(f)
        if not dt or dt.date() != target_date:
            continue
        # Cek apakah timestamp ini dipicu oleh sensor yang kita pilih
        ts_str = dt.strftime("%Y%m%d_%H%M%S")
        if ts_str in valid_timestamps:
            abn_files.append(f)
        # Jika tidak ada di valid_timestamps → sensor trigger bukan selected → skip

    n_skipped_total = sum(skipped_by_sensor.values())

    if verbose:
        split = get_split(target_date)
        print(f"  [{date_str}] {len(raw_files):3d} raw + "
              f"{len(abn_files):2d} ALL_TR (relevan) | "
              f"workers={n_workers} | split={split}")
        if skipped_by_sensor:
            # Tampilkan sensor mana yang paling banyak di-skip
            top = sorted(skipped_by_sensor.items(), key=lambda x:-x[1])[:3]
            top_str = ", ".join(f"{s}={n}" for s,n in top)
            print(f"           ↳ {n_skipped_total} event dari sensor excluded diskip "
                  f"[top: {top_str}]")

    if not raw_files and not abn_files:
        if verbose:
            print(f"  [{date_str}] Tidak ada file")
        return {"date": date_str, "status": "no_data",
                "n_raw": 0, "n_abnormal_files": 0, "n_windows": 0}

    # ── Pipeline: baca paralel → windowing segera ─────────
    # Setiap future mewakili satu file yang sedang/sudah dibaca.
    # Begitu satu file selesai dibaca, langsung di-windowing.
    # RAM yang terpakai: hanya n_workers file sekaligus, bukan 144 file.

    tasks = ([(f, False) for f in raw_files] +
             [(f, True)  for f in abn_files])

    all_X, all_y = [], []
    n_loaded = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # Submit semua task sekaligus
        futures = {
            ex.submit(load_one_file, fp, flag): (fp, flag)
            for fp, flag in tasks
        }

        # Proses hasil begitu ada yang selesai (as_completed = FIFO by readiness)
        for fut in as_completed(futures):
            arr = fut.result()
            if arr is None:
                continue

            fp, flag = futures[fut]
            if not flag:
                n_loaded += 1

            # Windowing langsung, jangan tunggu file lain
            X_chunk, y_chunk = make_windows(arr, _USE_IDX)
            if len(X_chunk) > 0:
                all_X.append(X_chunk)
                all_y.append(y_chunk)

            # Bebaskan array mentah segera
            del arr
            # gc per file tidak perlu, nanti gc bersama setelah loop

    if not all_X:
        return {"date": date_str, "status": "no_windows",
                "n_raw": n_loaded, "n_windows": 0}

    # ── Gabung semua chunk window (jauh lebih kecil dari raw data) ──
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    del all_X, all_y
    gc.collect()

    np.save(x_path, X)
    np.save(y_path, y)

    n_abn   = int(y.sum())
    size_mb = round((X.nbytes + y.nbytes) / 1024**2, 1)

    meta_row = {
        "date"              : date_str,
        "split"             : get_split(target_date),
        "status"            : "ok",
        "n_raw_files"       : len(raw_files),
        "n_raw_loaded"      : n_loaded,
        "n_abnormal_files"  : len(abn_files),
        "n_abn_skipped"     : n_skipped_total,   # event dari sensor excluded
        "n_windows"         : len(X),
        "n_windows_normal"  : len(X) - n_abn,
        "n_windows_abnormal": n_abn,
        "shape_X"           : str(X.shape),
        "size_mb"           : size_mb,
    }
    pd.DataFrame([meta_row]).to_csv(meta_path, index=False)

    if verbose:
        print(f"           → {len(X):,} windows "
              f"({n_abn} abnormal) | {size_mb} MB")

    del X, y
    gc.collect()

    return meta_row


# ─────────────────────────────────────────────────────────
# PROSES SEMUA HARI
# ─────────────────────────────────────────────────────────

def process_all(raw_dir: Path, abnormal_dir: Path, out_dir: Path,
                start: date = DATA_START, end: date = DATA_END,
                n_workers: int = DEFAULT_WORKERS):
    """Proses semua hari dari start sampai end secara berurutan."""
    print("\n" + "="*65)
    print("  SHMS BATCH PROCESSOR (optimized)")
    print(f"  Periode  : {start} s/d {end}")
    print(f"  Workers  : {n_workers} thread paralel per hari")
    print(f"  Optimasi : usecols, float32, pipeline I/O+CPU, pre-alloc")
    print("="*65)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_meta = []
    current  = start

    while current <= end:
        meta = process_one_day(current, raw_dir, abnormal_dir,
                               out_dir, n_workers=n_workers)
        all_meta.append(meta)
        current += timedelta(days=1)

    summary = pd.DataFrame(all_meta)
    summary.to_csv(out_dir / "processing_summary.csv", index=False)

    ok          = summary[summary["status"] == "ok"]
    n_win       = ok["n_windows"].sum() if len(ok) else 0
    n_abn_win   = ok["n_windows_abnormal"].sum() if "n_windows_abnormal" in ok.columns else 0
    n_abn_files = ok["n_abnormal_files"].sum() if "n_abnormal_files" in ok.columns else 0

    print("\n" + "="*65)
    print("  SELESAI")
    print(f"  Hari diproses      : {len(ok)} / {len(summary)}")
    print(f"  Total windows      : {n_win:,}")
    print(f"  Windows normal     : {n_win - n_abn_win:,}")
    print(f"  Windows abnormal   : {n_abn_win:,} "
          f"({100*n_abn_win/max(n_win,1):.2f}%)")
    print(f"  File ALL_TR dipakai: {int(n_abn_files):,} "
          f"(file sensor-spesifik dilewati otomatis)")
    if "size_mb" in ok.columns:
        total_mb = ok["size_mb"].sum()
        unit = "GB" if total_mb > 1024 else "MB"
        val  = total_mb/1024 if total_mb > 1024 else total_mb
        print(f"  Total .npy size    : {val:.1f} {unit}")
    print("\n  Per split:")
    for split in ["train", "val", "test"]:
        s = ok[ok["split"] == split]
        if len(s):
            print(f"    {split:5s}: {len(s):2d} hari | "
                  f"{s['n_windows'].sum():,} windows")
    print("="*65)


# ─────────────────────────────────────────────────────────
# CHECK PROGRESS
# ─────────────────────────────────────────────────────────

def check_progress(out_dir: Path):
    summary_path = out_dir / "processing_summary.csv"
    if not summary_path.exists():
        print("  Belum ada data yang diproses.")
        return
    df = pd.read_csv(summary_path)
    ok = df[df["status"] == "ok"]
    print(f"\n  Sudah diproses : {len(ok)} hari")
    print(f"  Total windows  : {ok['n_windows'].sum():,}")
    if "size_mb" in ok.columns:
        print(f"  Total size     : {ok['size_mb'].sum()/1024:.2f} GB")
    print(f"\n  Per split:")
    for split in ["train", "val", "test"]:
        s = ok[ok["split"] == split]
        print(f"    {split:5s}: {len(s):2d} hari | {s['n_windows'].sum():,} windows")


# ─────────────────────────────────────────────────────────
# LOAD PROCESSED SPLIT (untuk Phase 3-5)
# ─────────────────────────────────────────────────────────

def load_processed_split(out_dir: Path,
                         split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load semua .npy dari satu split ke dalam satu array."""
    summary_path = out_dir / "processing_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Jalankan batch processor dulu.")

    summary = pd.read_csv(summary_path)
    days    = summary[
        (summary["split"] == split) & (summary["status"] == "ok")
    ]["date"].tolist()

    X_list, y_list = [], []
    for d in sorted(days):
        xp = out_dir / f"{d}_X.npy"
        yp = out_dir / f"{d}_y.npy"
        if xp.exists():
            X_list.append(np.load(xp))
            y_list.append(np.load(yp))

    if not X_list:
        raise ValueError(f"Tidak ada data untuk split '{split}'")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"  [{split}] {len(X):,} windows "
          f"({y.sum()} abnormal, {100*y.mean():.1f}%)")
    return X, y


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SHMS Batch Processor — raw .txt → .npy (optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python shms_batch_processor.py                     # proses semua hari
  python shms_batch_processor.py --date 2026-02-06   # proses 1 hari saja
  python shms_batch_processor.py --check             # cek progress tanpa proses
  python shms_batch_processor.py --workers 4         # 4 thread paralel

Tips pemilihan --workers:
  HD USB 3.0 (seperti setup saat ini) : 2-3  (default)
  HD eksternal SSD                    : 4-6
  NVMe internal                       : 6-8
  Tanda terlalu banyak workers: Disk usage turun, CPU naik tinggi → kurangi
  Tanda terlalu sedikit workers: CPU idle, Disk 95%+ → naikkan (max 4 untuk USB)
        """
    )
    parser.add_argument("--date",    help="Proses 1 hari. Format: YYYY-MM-DD")
    parser.add_argument("--start",   help="Tanggal mulai. Format: YYYY-MM-DD")
    parser.add_argument("--end",     help="Tanggal akhir. Format: YYYY-MM-DD")
    parser.add_argument("--check",   action="store_true",
                        help="Tampilkan progress saja, tidak memproses")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Jumlah thread paralel (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    if args.check:
        check_progress(DATA_PROCESSED_DIR)
        return

    raw_dir = get_raw_dir()
    abn_dir = get_abnormal_dir()

    print(f"\n  Raw data    : {raw_dir}")
    print(f"  Abnormal    : {abn_dir}")
    print(f"  Output .npy : {DATA_PROCESSED_DIR}")
    print(f"  Workers     : {args.workers} thread")

    if args.date:
        target = date.fromisoformat(args.date)
        process_one_day(target, raw_dir, abn_dir, DATA_PROCESSED_DIR,
                        n_workers=args.workers)
        return

    start = date.fromisoformat(args.start) if args.start else DATA_START
    end   = date.fromisoformat(args.end)   if args.end   else DATA_END
    process_all(raw_dir, abn_dir, DATA_PROCESSED_DIR,
                start=start, end=end, n_workers=args.workers)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────
# FILTER SENSOR TRIGGER (tambahan untuk filter label noise)
# ─────────────────────────────────────────────────────────

def get_trigger_sensor(sensor_specific_path: Path) -> str | None:
    """
    Ekstrak nama sensor trigger dari file sensor-spesifik.

    Format nama file: TData__FB_CA_R27_20260206_101144.txt
    Output          : "FB_CA_R27"

    Cara: ambil semua bagian nama sebelum tanggal 8 digit (YYYYMMDD).
    """
    name  = sensor_specific_path.stem
    parts = name.replace(SENSOR_SPECIFIC_PREFIX, "").split("_")
    sensor_parts = []
    for part in parts:
        if len(part) == 8 and part.isdigit():   # YYYYMMDD → stop
            break
        if part:
            sensor_parts.append(part)
    return "_".join(sensor_parts) if sensor_parts else None


def is_trigger_in_selected(trigger_sensor: str) -> bool:
    """
    Apakah sensor yang memicu event termasuk dalam 20 channel model?

    Menggunakan partial match karena nama sensor trigger (mis. "FB_CA_L17")
    mungkin berbeda dengan nama channel DAQ (mis. "FB_CA_L17_EZ").

    Return True  → event relevan untuk model, ambil ALL_TR-nya
    Return False → event dari sensor rusak/tidak dipilih, skip ALL_TR-nya

    Contoh:
      "FB_CA_R27" → tidak ada match di ALL_CHANNELS → False (4182 event di-skip)
      "FB_CA_L17" → match "FB_CA_L17_EZ"            → True  (event diambil)
      "FB_DS_A1"  → strain gauge, tidak ada di model → False (di-skip)
    """
    if not trigger_sensor:
        return False
    for ch in ALL_CHANNELS:
        if trigger_sensor in ch or ch.replace("_EZ","").replace("_BX","") \
           .replace("_BY","").replace("_BZ","").replace("_S","") == trigger_sensor:
            return True
    return False


def get_valid_abn_timestamps(abnormal_dir: Path, target_date: date) -> set:
    """
    Scan semua file sensor-spesifik (TData__*) di abnormal_dir untuk
    tanggal target, dan kembalikan SET TIMESTAMP yang sensor trigger-nya
    ada di selected channels.

    Hanya timestamp dalam set ini yang boleh diambil ALL_TR-nya.

    Ini adalah kunci filter: dari 4182 event R27, semuanya akan
    memiliki timestamp yang TIDAK masuk set ini → semua di-skip.

    Return: set of string timestamp "YYYYMMDD_HHMMSS"
    """
    valid_ts = set()
    skipped_by_sensor = {}   # {sensor_name: count} untuk logging

    for f in abnormal_dir.glob("TData__*.txt"):
        # Ambil tanggal dari nama file sensor-spesifik
        stem  = f.stem  # "TData__FB_CA_R27_20260206_101144"
        parts = stem.replace(SENSOR_SPECIFIC_PREFIX, "").split("_")

        # Cari posisi tanggal 8 digit
        date_idx = None
        for i, p in enumerate(parts):
            if len(p) == 8 and p.isdigit():
                date_idx = i
                break

        if date_idx is None:
            continue

        # Parse tanggal
        try:
            ts_str = parts[date_idx] + "_" + parts[date_idx+1]
            dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            continue

        if dt.date() != target_date:
            continue

        # Ekstrak nama sensor trigger
        trigger = "_".join(parts[:date_idx]) if date_idx else None

        if is_trigger_in_selected(trigger):
            valid_ts.add(ts_str)
        else:
            skipped_by_sensor[trigger] = skipped_by_sensor.get(trigger, 0) + 1

    return valid_ts, skipped_by_sensor