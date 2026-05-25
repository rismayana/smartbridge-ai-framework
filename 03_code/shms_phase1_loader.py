"""
SHMS Bridge Anomaly Detection
Phase 1: Data Loader & EDA
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from shms_config import (
    ALL_CHANNELS, CHANNEL_ALIAS, ACCEL_CHANNELS, CABLE_CHANNELS,
    TEMP_CHANNELS, WIND_CHANNELS, CONTEXT_CHANNELS,
    EXCLUDED_CHANNELS, OUTPUT_DIR, SENSOR_TYPE_MAP
)


# ─────────────────────────────────────────────────
# 1A. LOADER
# ─────────────────────────────────────────────────

def load_csv(filepath, is_abnormal=False):
    """Load satu file CSV DAQ. Handle encoding latin-1 & mixed NaN."""
    df = pd.read_csv(filepath, encoding='latin-1',
                     na_values=['NaN', ' ', ''])
    # Bersihkan nama kolom
    df.columns = [c.strip().encode('ascii','ignore').decode('ascii').strip()
                  for c in df.columns]
    df = df.rename(columns={df.columns[0]: 'timestamp_raw'})

    # Convert sensor ke numeric
    sensor_cols = [c for c in df.columns if c.startswith('FB_')]
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['_source']     = Path(filepath).stem
    df['_is_abnormal'] = is_abnormal
    return df


def load_folder(folder, is_abnormal=False, pattern='*.csv'):
    """Load semua CSV dari folder."""
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"Tidak ada CSV di: {folder}")
    print(f"  Loading {len(files)} file dari {folder}...")
    dfs = [load_csv(f, is_abnormal) for f in files]
    out = pd.concat(dfs, ignore_index=True)
    print(f"  → {len(out):,} rows")
    return out


def load_stat_postgres(config, table='sensor_statistics'):
    """Load statistik 10-menit dari PostgreSQL."""
    try:
        from sqlalchemy import create_engine
        engine = create_engine(
            f"postgresql+psycopg2://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )
        df = pd.read_sql(f"SELECT * FROM {table} ORDER BY timestamp ASC", engine)
        print(f"  PostgreSQL: {len(df):,} rows dari tabel '{table}'")
        return df
    except Exception as e:
        print(f"  [WARNING] PostgreSQL gagal: {e}")
        return None


# ─────────────────────────────────────────────────
# 1B. SENSOR HEALTH CHECK
# ─────────────────────────────────────────────────

def check_sensor_health(df):
    """
    Diagnosis otomatis semua channel:
    - ALL NaN
    - Flatline (std = 0)
    - Offset tidak wajar (accelerometer > 500 Gal)
    - Nilai negatif (cable tension)
    - Near-zero (cable tidak terkoneksi)
    """
    sensor_cols = [c for c in df.columns if c.startswith('FB_')]
    results = []

    for col in sensor_cols:
        v    = df[col].dropna()
        miss = df[col].isna().sum() / len(df) * 100
        code = col.split('_')[1]
        stype = SENSOR_TYPE_MAP.get(code, code)

        if len(v) == 0:
            status, detail = 'ALL_NAN', 'semua nilai kosong'
        elif v.std() == 0:
            status, detail = 'FLATLINE', f'std=0, nilai={v.mean():.3f}'
        elif stype == 'Accelerometer' and abs(v.mean()) > 500:
            status, detail = 'OFFSET', f'mean={v.mean():.1f} Gal'
        elif stype == 'Cable Tensionmeter' and v.mean() < -100:
            status, detail = 'NEGATIVE', f'mean={v.mean():.1f} kN'
        elif stype == 'Cable Tensionmeter' and abs(v.mean()) < 20:
            status, detail = 'NEAR_ZERO', f'mean={v.mean():.2f} kN'
        elif v.std() < 1e-4 and len(v) > 10:
            status, detail = 'NEAR_FLAT', f'std={v.std():.2e}'
        else:
            status, detail = 'OK', ''

        results.append({
            'channel'    : col,
            'alias'      : CHANNEL_ALIAS.get(col, '—'),
            'type'       : stype,
            'status'     : status,
            'detail'     : detail,
            'mean'       : round(v.mean(), 4) if len(v) > 0 else np.nan,
            'std'        : round(v.std(),  4) if len(v) > 0 else np.nan,
            'missing_pct': round(miss, 1),
            'is_selected': col in ALL_CHANNELS,
            'is_excluded': col in EXCLUDED_CHANNELS,
        })

    return pd.DataFrame(results)


def print_health_summary(health_df):
    print("\n" + "="*65)
    print("  SENSOR HEALTH SUMMARY")
    print("="*65)
    total = len(health_df)
    ok    = (health_df['status'] == 'OK').sum()
    print(f"  Total channel : {total}")
    print(f"  OK            : {ok}")
    print(f"  Bermasalah    : {total - ok}")

    print("\n  Channel TERPILIH (20 channel final):")
    sel = health_df[health_df['is_selected']]
    for _, r in sel.iterrows():
        flag = f"  ← {r['detail']}" if r['status'] != 'OK' else ''
        print(f"    [{r['status']:10s}] {r['alias']:15s} {r['channel']}{flag}")

    prob = health_df[(health_df['status'] != 'OK') & (~health_df['is_selected'])]
    if len(prob):
        print(f"\n  Channel BERMASALAH (tidak dipilih): {len(prob)}")
        for _, r in prob.iterrows():
            print(f"    [{r['status']:10s}] {r['channel']:35s} {r['detail']}")


# ─────────────────────────────────────────────────
# 1C. EXTRACT & RESAMPLE
# ─────────────────────────────────────────────────

def extract_selected(df):
    """
    Ambil 20 channel terpilih.
    Anemometer (1Hz): forward-fill ke seluruh baris.
    """
    available = [c for c in ALL_CHANNELS if c in df.columns]
    missing   = [c for c in ALL_CHANNELS if c not in df.columns]
    if missing:
        print(f"  [WARNING] Channel tidak ada di data: {missing}")

    meta = ['timestamp_raw', '_source', '_is_abnormal']
    meta = [c for c in meta if c in df.columns]

    out = df[meta + available].copy()

    # Forward-fill anemometer (1Hz → ikuti sampling rate sensor lain)
    for col in WIND_CHANNELS:
        if col in out.columns:
            out[col] = out[col].ffill().bfill()

    print(f"  Dataset terpilih: {len(out):,} rows × {len(available)} channel")
    return out


# ─────────────────────────────────────────────────
# 1D. EDA PLOTS
# ─────────────────────────────────────────────────

def plot_eda(df_sel, save_dir=OUTPUT_DIR):
    save_dir = Path(save_dir)
    sensor_cols = [c for c in df_sel.columns if c in ALL_CHANNELS]

    # ── Plot 1: Correlation heatmap ──────────────
    corr        = df_sel[sensor_cols].corr()
    short_names = [CHANNEL_ALIAS.get(c, c.replace('FB_','')) for c in sensor_cols]

    n = len(sensor_cols)
    fig, ax = plt.subplots(figsize=(max(12, n), max(10, n-2)))
    mask = np.eye(len(corr), dtype=bool)
    sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.3, annot=True, fmt='.2f',
                annot_kws={'size': 8}, mask=mask, square=True)
    ax.set_title('Correlation matrix — 20 channel terpilih\nSHMS Finger Bridge',
                 fontsize=13, pad=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_dir / 'eda_01_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Time-series per group ────────────
    groups = {
        'Accelerometer (Gal)' : ACCEL_CHANNELS,
        'Cable Tension (kN)'  : CABLE_CHANNELS,
        'Temperature (°C)'    : TEMP_CHANNELS,
        'Wind Speed (m/s)'    : WIND_CHANNELS,
    }
    colors = {'Accelerometer (Gal)': '#378ADD',
              'Cable Tension (kN)' : '#1D9E75',
              'Temperature (°C)'   : '#D85A30',
              'Wind Speed (m/s)'   : '#BA7517'}

    for grp_name, cols in groups.items():
        valid = [c for c in cols if c in df_sel.columns]
        if not valid:
            continue
        fig, axes = plt.subplots(len(valid), 1,
                                 figsize=(14, 2.5*len(valid)), sharex=True)
        if len(valid) == 1:
            axes = [axes]
        for ax, col in zip(axes, valid):
            alias = CHANNEL_ALIAS.get(col, col.replace('FB_',''))
            v     = df_sel[col]
            ax.plot(v.values, color=colors[grp_name], linewidth=0.9, alpha=0.85)
            ax.axhline(v.mean(), color='red', linewidth=0.7,
                       linestyle='--', alpha=0.7, label=f'mean={v.mean():.3f}')
            ax.set_ylabel(alias, fontsize=8, rotation=0,
                          ha='right', labelpad=80)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)
        axes[-1].set_xlabel('Sample index', fontsize=9)
        fig.suptitle(f'Time-series: {grp_name}', fontsize=12, y=1.002)
        plt.tight_layout()
        safe = grp_name.split()[0].lower()
        fig.savefig(save_dir / f'eda_02_ts_{safe}.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    # ── Plot 3: Distribution per channel ─────────
    n_cols_plot = min(5, len(sensor_cols))
    n_rows_plot = (len(sensor_cols) + n_cols_plot - 1) // n_cols_plot
    fig, axes   = plt.subplots(n_rows_plot, n_cols_plot,
                               figsize=(4*n_cols_plot, 3*n_rows_plot))
    axes_flat   = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, col in enumerate(sensor_cols):
        ax    = axes_flat[i]
        alias = CHANNEL_ALIAS.get(col, col.replace('FB_',''))
        v     = df_sel[col].dropna()
        ax.hist(v, bins=30, color='#378ADD', edgecolor='white', linewidth=0.4)
        ax.set_title(alias, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
    for j in range(len(sensor_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle('Distribusi nilai — 20 channel terpilih', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir / 'eda_03_distributions.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"\n  EDA plots disimpan di: {save_dir}/")
    print(f"    eda_01_correlation.png")
    print(f"    eda_02_ts_accelerometer.png")
    print(f"    eda_02_ts_cable.png")
    print(f"    eda_02_ts_temperature.png")
    print(f"    eda_03_distributions.png")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def run_phase1(raw_dir, abnormal_dir=None, pg_config=None):
    print("\n" + "="*65)
    print("  SHMS ANOMALY DETECTION — PHASE 1")
    print("="*65)

    # Load
    df_raw = load_folder(raw_dir, is_abnormal=False)
    if abnormal_dir and Path(abnormal_dir).exists():
        df_abn = load_folder(abnormal_dir, is_abnormal=True)
        df_all = pd.concat([df_raw, df_abn], ignore_index=True)
    else:
        df_all = df_raw

    df_stat = load_stat_postgres(pg_config) if pg_config else None

    # Health check
    health = check_sensor_health(df_all)
    print_health_summary(health)
    health.to_csv(OUTPUT_DIR / 'p1_sensor_health.csv', index=False)

    # Extract
    df_sel = extract_selected(df_all)
    df_sel.to_csv(OUTPUT_DIR / 'p1_dataset_selected.csv', index=False)

    # EDA
    print("\n  Membuat EDA plots...")
    plot_eda(df_sel)

    print("\n" + "="*65)
    print("  PHASE 1 SELESAI → lanjut Phase 2: Preprocessing")
    print("="*65)
    return df_sel, df_stat, health


# Quick test satu file
def quicktest(filepath):
    print(f"\n[QUICKTEST] {filepath}")
    df  = load_csv(filepath)
    h   = check_sensor_health(df)
    print_health_summary(h)
    sel = extract_selected(df)
    plot_eda(sel)
    return sel, h


if __name__ == '__main__':
    sel, h = quicktest('data/raw/SAMPLE_20260329234000.csv')
