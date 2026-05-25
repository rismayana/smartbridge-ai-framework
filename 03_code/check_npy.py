"""
check_npy.py — Diagnostic tool untuk verifikasi semua file .npy

Cara pakai:
    python check_npy.py             # cek semua, tampilkan yang bermasalah
    python check_npy.py --fix       # otomatis hapus file corrupt
    python check_npy.py --verbose   # tampilkan semua file (tidak hanya yang bermasalah)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from shms_config import get_processed_dir

def main():
    parser = argparse.ArgumentParser(description="Cek integritas file .npy")
    parser.add_argument("--fix",     action="store_true",
                        help="Hapus file corrupt + update summary.csv")
    parser.add_argument("--verbose", action="store_true",
                        help="Tampilkan semua file, bukan hanya yang bermasalah")
    args = parser.parse_args()

    out_dir = get_processed_dir()
    print(f"\n  Folder .npy : {out_dir}")
    print("=" * 65)

    summary_path = out_dir / "processing_summary.csv"
    if not summary_path.exists():
        print("  ERROR: processing_summary.csv tidak ditemukan!")
        print(f"  Pastikan batch processor sudah dijalankan.")
        sys.exit(1)

    summary  = pd.read_csv(summary_path)
    ok_days  = summary[summary["status"] == "ok"]["date"].tolist()
    print(f"  Hari terdaftar OK di summary: {len(ok_days)}")
    print()

    corrupt, missing = [], []
    ok_count = 0
    total_size_gb = 0

    print(f"  {'Tanggal':<12} {'Status':<10} {'Size (MB)':>10} {'Windows':>10}  {'Split'}")
    print(f"  {'-'*58}")

    for d in sorted(ok_days):
        xp = out_dir / f"{d}_X.npy"
        yp = out_dir / f"{d}_y.npy"

        # Cek split
        split = summary[summary["date"]==int(d)]["split"].values
        split = split[0] if len(split) else "?"

        if not xp.exists():
            if args.verbose or True:
                print(f"  {d:<12} {'MISSING':<10} {'N/A':>10} {'N/A':>10}  {split}")
            missing.append(d)
            continue

        size_mb = xp.stat().st_size / 1024**2
        total_size_gb += size_mb / 1024

        # Baca header dengan mmap (tidak load ke RAM)
        try:
            arr = np.load(str(xp), mmap_mode="r")
            if len(arr.shape) != 3:
                raise ValueError(f"Shape salah: {arr.shape}")
            n_windows = arr.shape[0]
            del arr  # release mmap

            if args.verbose:
                print(f"  {d:<12} {'OK':<10} {size_mb:>10.0f} {n_windows:>10,}  {split}")
            ok_count += 1

        except Exception as e:
            print(f"  {d:<12} {'CORRUPT':<10} {size_mb:>10.1f} {'ERROR':>10}  {split}")
            print(f"             └→ {str(e)[:60]}")
            corrupt.append(d)

    # Ringkasan
    print()
    print("=" * 65)
    print(f"  RINGKASAN:")
    print(f"    OK      : {ok_count} hari")
    print(f"    Corrupt : {len(corrupt)} hari")
    print(f"    Missing : {len(missing)} hari")
    print(f"    Total ukuran: {total_size_gb:.1f} GB")

    if not corrupt and not missing:
        print(f"\n  ✅ Semua file .npy dalam kondisi baik!")
        print(f"  ✅ Phase 3A siap dijalankan.")
        return

    # Solusi
    print(f"\n  MASALAH YANG DITEMUKAN:")
    if missing:
        print(f"  File tidak ada (kemungkinan belum diproses):")
        for d in missing:
            print(f"    - {d}")
    if corrupt:
        print(f"  File corrupt (tulis tidak lengkap):")
        for d in corrupt:
            print(f"    - {d}_X.npy")

    print(f"""
  CARA MEMPERBAIKI:

  Opsi A — Jalankan ulang batch processor untuk hari bermasalah:
    (batch processor otomatis skip hari yang sudah OK)""")

    all_bad = list(set(corrupt + missing))
    for d in sorted(all_bad):
        date_fmt = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        print(f"    python shms_batch_processor.py --date {date_fmt}")

    print(f"""
  Opsi B — Jalankan ulang batch processor semua hari:
    (akan skip hari yang sudah OK, proses ulang yang bermasalah)
    python shms_batch_processor.py

  Opsi C — Fix otomatis (hapus file corrupt, lanjut batch):""")
    if args.fix:
        print(f"\n  Menjalankan --fix...")
        fixed = 0
        for d in corrupt:
            for suffix in ["_X.npy", "_y.npy", "_meta.csv"]:
                f = out_dir / f"{d}{suffix}"
                if f.exists():
                    f.unlink()
                    print(f"    Dihapus: {f.name}")

            # Update summary.csv
            summary.loc[summary["date"]==int(d), "status"] = "failed_corrupt"
            fixed += 1

        if fixed > 0:
            summary.to_csv(summary_path, index=False)
            print(f"\n  ✅ {fixed} hari corrupt dihapus dan ditandai di summary.csv")
            print(f"  Sekarang jalankan:")
            print(f"    python shms_batch_processor.py")
    else:
        print(f"    python check_npy.py --fix")
        print(f"    python shms_batch_processor.py")

if __name__ == "__main__":
    main()