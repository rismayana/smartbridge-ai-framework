"""
shms_check_nov2025.py — Scan semua file .npy November 2025
Identifikasi file corrupt, kosong, atau ukuran tidak wajar.

Cara pakai:
  # Scan semua file (hanya baca header, tidak load ke RAM):
  python shms_check_nov2025.py

  # Setelah menemukan file corrupt, proses ulang hari tersebut:
  python shms_batch_processor_nov2025.py --date 2025-11-XX --force
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))
from shms_config import get_processed_dir

def check_npy_file(path: Path) -> dict:
    """
    Validasi file .npy tanpa load data penuh ke RAM.
    Hanya baca header untuk cek shape dan dtype.
    """
    result = {
        "path"    : path,
        "size_mb" : round(path.stat().st_size / 1024**2, 1),
        "status"  : "unknown",
        "shape"   : None,
        "n_windows": 0,
        "note"    : "",
    }

    if result["size_mb"] == 0:
        result["status"] = "EMPTY"
        result["note"]   = "File 0 bytes — batch processor terhenti saat menulis"
        return result

    try:
        # Baca header saja (tidak load array ke RAM)
        with open(path, "rb") as f:
            # Magic string numpy: \x93NUMPY
            magic = f.read(6)
            if magic[:6] != b'\x93NUMPY':
                result["status"] = "NOT_NPY"
                result["note"]   = "Bukan file numpy valid"
                return result

            # Baca header untuk dapat shape
            major = f.read(1)[0]
            minor = f.read(1)[0]
            if major == 1:
                header_len = int.from_bytes(f.read(2), 'little')
            else:
                header_len = int.from_bytes(f.read(4), 'little')

            header = f.read(header_len).decode('latin1')

        # Parse shape dari header string
        import ast, re
        shape_match = re.search(r"'shape'\s*:\s*(\([^)]*\))", header)
        if shape_match:
            shape = ast.literal_eval(shape_match.group(1))
            result["shape"]    = shape
            result["n_windows"] = shape[0] if shape else 0

            # Validasi shape: harus (n, 1000, 20)
            if len(shape) != 3:
                result["status"] = "WRONG_DIM"
                result["note"]   = f"Dimensi salah: {len(shape)}D (harusnya 3D)"
            elif shape[0] == 0:
                result["status"] = "ZERO_WINDOWS"
                result["note"]   = "0 windows — file corrupt saat penulisan"
            elif shape[1] != 1000:
                result["status"] = "WRONG_WINSIZE"
                result["note"]   = f"Window size {shape[1]} (harusnya 1000)"
            elif shape[2] not in (17, 20):
                result["status"] = "WRONG_CHANNELS"
                result["note"]   = f"Channel {shape[2]} (harusnya 17 atau 20)"
            else:
                # Estimasi ukuran yang diharapkan
                expected_mb = shape[0] * shape[1] * shape[2] * 4 / 1024**2
                ratio = result["size_mb"] / expected_mb if expected_mb > 0 else 0
                if ratio < 0.8:
                    result["status"] = "TRUNCATED"
                    result["note"]   = (f"File terpotong: "
                                        f"{result['size_mb']:.0f} MB "
                                        f"vs ekspektasi {expected_mb:.0f} MB "
                                        f"({ratio*100:.0f}%)")
                else:
                    result["status"] = "OK"
        else:
            result["status"] = "PARSE_ERROR"
            result["note"]   = "Tidak bisa parse shape dari header"

    except Exception as e:
        result["status"] = "READ_ERROR"
        result["note"]   = str(e)

    return result


def main():
    processed = get_processed_dir() / "nov2025"

    print("\n" + "="*65)
    print("  DIAGNOSTIC — Scan file .npy November 2025")
    print(f"  Folder: {processed}")
    print("="*65 + "\n")

    if not processed.exists():
        print("  ❌ Folder tidak ditemukan.")
        print("     Jalankan shms_batch_processor_nov2025.py dulu.")
        return

    npy_files = sorted(processed.glob("*_X.npy"))
    if not npy_files:
        print("  Tidak ada file *_X.npy ditemukan.")
        return

    print(f"  Total file ditemukan: {len(npy_files)}\n")
    print(f"  {'Tanggal':<12} {'Status':<14} {'Size(MB)':>9} "
          f"{'Windows':>9} {'Shape':<20} Catatan")
    print(f"  {'-'*80}")

    results  = []
    n_ok     = 0
    n_bad    = 0
    bad_days = []

    for fpath in npy_files:
        r   = check_npy_file(fpath)
        date_str = fpath.stem.replace("_X", "")

        is_ok  = r["status"] == "OK"
        icon   = "✅" if is_ok else "❌"
        shape_s = str(r["shape"]) if r["shape"] else "-"

        print(f"  {icon} {date_str:<10} {r['status']:<14} "
              f"{r['size_mb']:>9.1f} {r['n_windows']:>9,} "
              f"{shape_s:<20} {r['note']}")

        if is_ok:
            n_ok += 1
        else:
            n_bad += 1
            bad_days.append(date_str)

        results.append({
            "date"     : date_str,
            "status"   : r["status"],
            "size_mb"  : r["size_mb"],
            "n_windows": r["n_windows"],
            "shape"    : str(r["shape"]),
            "note"     : r["note"],
        })

    # Simpan laporan
    report_path = processed / "diagnostic_report.csv"
    pd.DataFrame(results).to_csv(report_path, index=False)

    print(f"\n  {'='*65}")
    print(f"  RINGKASAN:")
    print(f"  File OK      : {n_ok}")
    print(f"  File CORRUPT : {n_bad}")
    print(f"  Report       : {report_path}")

    if bad_days:
        print(f"\n  ⚠️  File corrupt ditemukan pada hari:")
        for d in bad_days:
            y, m, day = int(d[:4]), int(d[4:6]), int(d[6:8])
            print(f"    python shms_batch_processor_nov2025.py "
                  f"--date {y}-{m:02d}-{day:02d} --force")

    print(f"\n  Setelah repair, jalankan ulang OOD validation:")
    print(f"  python shms_ood_validation_nov2025.py")
    print("="*65)


if __name__ == "__main__":
    main()
