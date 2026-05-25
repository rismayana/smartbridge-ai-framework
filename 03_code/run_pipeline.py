"""
SHMS Bridge Anomaly Detection
run_pipeline.py — entry point, jalankan semua phase sekaligus

Cara pakai:
    python run_pipeline.py                    # Phase 1 & 2 (default)
    python run_pipeline.py --phase all        # semua phase
    python run_pipeline.py --phase 1          # hanya Phase 1
    python run_pipeline.py --phase 1 2        # Phase 1 dan 2
    python run_pipeline.py --phase 3a 3b 3c   # semua model
    python run_pipeline.py --phase batch      # batch processor (untuk data besar)
    python run_pipeline.py --retrain          # paksa retrain meski model sudah ada

Catatan HD Portable:
    Set RAW_DATA_OVERRIDE di shms_config.py jika data raw di HD eksternal.
    Phase 1, 2, dan batch membutuhkan akses ke data raw.
    Phase 3-5 hanya butuh data processed (.npy) di DATA_PROCESSED_DIR.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shms_config import (
    DATA_PROCESSED_DIR, MODEL_DIR, RESULTS_DIR,
    get_raw_dir, get_abnormal_dir
)


def run(phases: list, retrain: bool = False):
    results = {}
    t_start = time.time()
    all_ok  = True

    # ── Batch Processor ──────────────────────────────────────────
    if 'batch' in phases:
        print("\n" + "="*60)
        print("  BATCH PROCESSOR — Proses data raw per hari")
        print("="*60)
        try:
            from shms_batch_processor import process_all
            process_all(
                raw_dir      = get_raw_dir(),
                abnormal_dir = get_abnormal_dir(),
                out_dir      = DATA_PROCESSED_DIR,
            )
            print("  ✅ Batch processor selesai")
        except Exception as e:
            print(f"  ❌ Batch processor error: {e}")
            all_ok = False

    # ── Phase 1 ──────────────────────────────────────────────────
    if '1' in phases:
        print("\n" + "="*60)
        print("  PHASE 1 — Data Loading & EDA")
        print("="*60)
        try:
            from shms_phase1_loader import run_phase1
            df_sel, df_stat, health = run_phase1(
                raw_dir      = get_raw_dir(),
                abnormal_dir = get_abnormal_dir(),
            )
            results['df_sel'] = df_sel
            results['health'] = health
            print("  ✅ Phase 1 selesai")
        except Exception as e:
            print(f"  ❌ Phase 1 error: {e}")
            all_ok = False

    # ── Phase 2 ──────────────────────────────────────────────────
    if '2' in phases:
        print("\n" + "="*60)
        print("  PHASE 2 — Preprocessing")
        print("="*60)
        try:
            from shms_phase2_preprocessing import run_phase2
            df_sel = results.get('df_sel')
            if df_sel is None:
                print("  [INFO] Load Phase 1 output terlebih dahulu...")
                from shms_phase1_loader import run_phase1
                df_sel, _, _ = run_phase1(get_raw_dir(), get_abnormal_dir())
            out = run_phase2(df_sel)
            results.update({
                'X_train'   : out[0], 'X_test'    : out[1],
                'y_train'   : out[2], 'y_test'    : out[3],
                'norm'      : out[4], 'corr'      : out[5],
                'feat_train': out[6], 'edges'     : out[7],
            })
            print("  ✅ Phase 2 selesai")
        except Exception as e:
            print(f"  ❌ Phase 2 error: {e}")
            all_ok = False

    # ── Phase 3A: LSTM ───────────────────────────────────────────
    if '3a' in phases:
        print("\n" + "="*60)
        print("  PHASE 3A — LSTM Autoencoder")
        print("  [INFO] Rekomendasi: jalankan di Colab T4 GPU")
        print("  [INFO] Notebook: SHMS_Phase3A_LSTM_Autoencoder.ipynb")
        print("="*60)
        try:
            from shms_phase3a_lstm import run_phase3a
            metrics = run_phase3a(retrain=retrain)
            results['lstm_metrics'] = metrics
            print(f"  ✅ Phase 3A selesai | F1={metrics.get('f1',0):.4f}")
        except ImportError as e:
            print(f"  ⚠️  Dependency tidak tersedia: {e}")
            print(f"  → Install: pip install torch")
            print(f"  → Atau gunakan Google Colab (T4 GPU)")
        except Exception as e:
            print(f"  ❌ Phase 3A error: {e}")
            all_ok = False

    # ── Phase 3B: Isolation Forest ───────────────────────────────
    if '3b' in phases:
        print("\n" + "="*60)
        print("  PHASE 3B — Isolation Forest")
        print("="*60)
        try:
            from shms_phase3b_iforest import run_phase3b
            metrics = run_phase3b(retrain=retrain)
            results['iforest_metrics'] = metrics
            print(f"  ✅ Phase 3B selesai | F1={metrics.get('f1',0):.4f}")
        except Exception as e:
            print(f"  ❌ Phase 3B error: {e}")
            all_ok = False

    # ── Phase 3C: GNN ────────────────────────────────────────────
    if '3c' in phases:
        print("\n" + "="*60)
        print("  PHASE 3C — GNN Autoencoder")
        print("  [INFO] Rekomendasi: jalankan di Colab T4 GPU")
        print("  [INFO] Notebook: SHMS_Phase3C_GNN.ipynb")
        print("="*60)
        try:
            from shms_phase3c_gnn import run_phase3c
            metrics = run_phase3c(retrain=retrain)
            results['gnn_metrics'] = metrics
            print(f"  ✅ Phase 3C selesai | F1={metrics.get('f1',0):.4f}")
        except ImportError as e:
            print(f"  ⚠️  Dependency tidak tersedia: {e}")
            print(f"  → Install: pip install torch torch_geometric")
            print(f"  → Atau gunakan Google Colab (T4 GPU)")
        except Exception as e:
            print(f"  ❌ Phase 3C error: {e}")
            all_ok = False

    # ── Phase 4: Ensemble ────────────────────────────────────────
    if '4' in phases:
        print("\n" + "="*60)
        print("  PHASE 4 — Ensemble Fusion")
        print("="*60)
        try:
            from shms_phase4_ensemble import run_phase4
            metrics = run_phase4(weight_method='f1')
            results['ensemble_metrics'] = metrics
            print(f"  ✅ Phase 4 selesai | F1={metrics.get('f1',0):.4f}")
        except Exception as e:
            print(f"  ❌ Phase 4 error: {e}")
            all_ok = False

    # ── Phase 5: Evaluation ──────────────────────────────────────
    if '5' in phases:
        print("\n" + "="*60)
        print("  PHASE 5 — Ablation Study & Evaluasi Final")
        print("="*60)
        try:
            from shms_phase5_evaluation import run_phase5
            out5 = run_phase5()
            results['evaluation'] = out5
            print("  ✅ Phase 5 selesai")
        except Exception as e:
            print(f"  ❌ Phase 5 error: {e}")
            all_ok = False

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print("  PIPELINE " + ("✅ SELESAI" if all_ok else "⚠️  SELESAI DENGAN ERROR"))
    print(f"  Total waktu: {elapsed:.0f}s ({elapsed/60:.1f} menit)")
    print("="*60)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='SHMS AI Anomaly Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Urutan eksekusi yang disarankan:
  1.  python run_pipeline.py --phase batch     # proses data raw (HD portable)
  2.  [Colab] SHMS_Phase3A_LSTM.ipynb         # LSTM (GPU)
  3.  python run_pipeline.py --phase 3b        # Isolation Forest (CPU)
  4.  [Colab] SHMS_Phase3C_GNN.ipynb          # GNN (GPU)
  5.  python run_pipeline.py --phase 4 5       # Ensemble + Evaluasi (CPU)
        """
    )
    parser.add_argument(
        '--phase', nargs='+',
        default=['1', '2'],
        choices=['1', '2', '3a', '3b', '3c', '4', '5', 'batch', 'all'],
        help='Phase yang dijalankan (default: 1 2)'
    )
    parser.add_argument(
        '--retrain', action='store_true',
        help='Paksa retrain meski model sudah ada'
    )
    args   = parser.parse_args()
    phases = args.phase
    if 'all' in phases:
        phases = ['1', '2', '3a', '3b', '3c', '4', '5']
    print(f"\nPhase: {phases}" + (" | RETRAIN=True" if args.retrain else ""))
    run(phases, retrain=args.retrain)


if __name__ == '__main__':
    main()
