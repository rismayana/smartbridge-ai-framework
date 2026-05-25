"""
SHMS Bridge Anomaly Detection
Phase 2: Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from shms_config import (
    ALL_CHANNELS, CHANNEL_ALIAS,
    ACCEL_CHANNELS, CABLE_CHANNELS, TEMP_CHANNELS, WIND_CHANNELS,
    CONTEXT_CHANNELS, OUTPUT_DIR, WINDOW_SIZE, WINDOW_STEP
)


# ─────────────────────────────────────────────────
# 2A. ADAPTIVE NORMALIZATION
# ─────────────────────────────────────────────────

class AdaptiveNormalizer:
    """
    Z-score normalization per channel, dihitung dari data training.
    Adaptive: stats disimpan dan bisa di-update secara incremental.

    Berbeda dengan static threshold sistem eksisting:
    - Tidak butuh nilai min/max manual
    - Bisa di-update saat distribusi data bergeser (konsep drift)
    """

    def __init__(self):
        self.stats_ = {}   # {channel: {'mean': x, 'std': x}}
        self.fitted_ = False

    def fit(self, df, cols=None):
        """Hitung mean & std dari data training."""
        cols = cols or [c for c in df.columns if c in ALL_CHANNELS]
        for col in cols:
            v = df[col].dropna()
            if len(v) < 2:
                print(f"  [SKIP] {col} — tidak cukup data untuk normalisasi")
                continue
            self.stats_[col] = {
                'mean': v.mean(),
                'std' : v.std() if v.std() > 1e-9 else 1.0  # hindari div/0
            }
        self.fitted_ = True
        print(f"  Normalizer fitted: {len(self.stats_)} channel")
        return self

    def transform(self, df):
        """Terapkan z-score: (x - mean) / std"""
        if not self.fitted_:
            raise RuntimeError("Panggil fit() dulu sebelum transform()")
        out = df.copy()
        for col, stat in self.stats_.items():
            if col in out.columns:
                out[col] = (out[col] - stat['mean']) / stat['std']
        return out

    def fit_transform(self, df, cols=None):
        return self.fit(df, cols).transform(df)

    def save(self, path):
        """Simpan stats ke CSV untuk reproducibility."""
        rows = [{'channel': k, **v} for k, v in self.stats_.items()]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Normalizer stats disimpan: {path}")

    @classmethod
    def load(cls, path):
        """Load stats dari file."""
        obj = cls()
        df  = pd.read_csv(path)
        obj.stats_  = {r['channel']: {'mean': r['mean'], 'std': r['std']}
                       for _, r in df.iterrows()}
        obj.fitted_ = True
        return obj


# ─────────────────────────────────────────────────
# 2B. CORRELATION MATRIX
# ─────────────────────────────────────────────────

def compute_correlation_matrix(df, cols=None, save=True):
    """
    Hitung correlation matrix antar channel terpilih.
    Digunakan untuk:
    1. Memahami hubungan fisik antar sensor
    2. Membangun graph edge untuk GNN (Phase 3)
    3. Mendeteksi pergeseran korelasi sebagai sinyal anomali
    """
    cols  = cols or [c for c in df.columns if c in ALL_CHANNELS
                     and c not in CONTEXT_CHANNELS]
    corr  = df[cols].corr()
    alias = [CHANNEL_ALIAS.get(c, c.replace('FB_','')) for c in cols]

    if save:
        # Simpan matrix numerik
        corr_renamed = corr.copy()
        corr_renamed.index   = alias
        corr_renamed.columns = alias
        corr_renamed.to_csv(OUTPUT_DIR / 'p2_correlation_matrix.csv')

        # Plot heatmap
        n   = len(cols)
        fig, ax = plt.subplots(figsize=(max(10, n), max(8, n-2)))
        import seaborn as sns
        mask = np.eye(len(corr), dtype=bool)
        sns.heatmap(corr.rename(index=CHANNEL_ALIAS, columns=CHANNEL_ALIAS),
                    ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    linewidths=0.3, annot=True, fmt='.2f',
                    annot_kws={'size': 8}, mask=mask, square=True)
        ax.set_title('Correlation matrix — channel utama (post-normalisasi)\n'
                     'Input graph untuk GNN module', fontsize=12, pad=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'p2_correlation_heatmap.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Correlation matrix: {n}×{n} disimpan")

    return corr


def build_graph_edges(corr_matrix, threshold=0.5):
    """
    Bangun daftar edge untuk GNN berdasarkan korelasi.
    Edge dibuat jika |corr| >= threshold.

    Returns:
        edges: list of (node_i, node_j, weight)
    """
    cols  = corr_matrix.columns.tolist()
    edges = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            w = corr_matrix.iloc[i, j]
            if abs(w) >= threshold:
                edges.append({
                    'node_i'   : CHANNEL_ALIAS.get(cols[i], cols[i]),
                    'node_j'   : CHANNEL_ALIAS.get(cols[j], cols[j]),
                    'channel_i': cols[i],
                    'channel_j': cols[j],
                    'weight'   : round(w, 4),
                })
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(OUTPUT_DIR / 'p2_graph_edges.csv', index=False)
    print(f"  Graph edges (|corr|≥{threshold}): {len(edges_df)} edges")
    return edges_df


# ─────────────────────────────────────────────────
# 2C. SLIDING WINDOW SEGMENTATION
# ─────────────────────────────────────────────────

def sliding_window(df, cols, window_size=WINDOW_SIZE,
                   step=WINDOW_STEP, label_col=None):
    """
    Potong time-series menjadi window-window overlapping.

    Args:
        df          : DataFrame hasil normalisasi
        cols        : channel yang di-window
        window_size : jumlah sampel per window (default 1000 = 10 detik @100Hz)
        step        : langkah antar window (default 500 = overlap 50%)
        label_col   : kolom label jika ada (untuk supervised)

    Returns:
        X : ndarray (n_windows, window_size, n_channels)
        y : ndarray (n_windows,) jika label_col tersedia, else None
        meta : list of dict metadata tiap window
    """
    data   = df[cols].values.astype(np.float32)
    n_rows = len(data)
    X, y, meta = [], [], []

    for start in range(0, n_rows - window_size + 1, step):
        end    = start + window_size
        window = data[start:end]

        # Skip window yang lebih dari 10% NaN
        nan_ratio = np.isnan(window).mean()
        if nan_ratio > 0.10:
            continue

        # Interpolasi NaN sisa dalam window
        window = pd.DataFrame(window).interpolate(
            method='linear', limit_direction='both').values

        X.append(window)

        if label_col and label_col in df.columns:
            # Label window = 1 jika lebih dari 50% baris dalam window adalah abnormal
            lbl = df[label_col].iloc[start:end].mode()[0]
            y.append(int(lbl))

        meta.append({
            'start_idx': start,
            'end_idx'  : end,
            'source'   : df['_source'].iloc[start] if '_source' in df.columns else '',
        })

    X = np.array(X)
    y = np.array(y) if y else None

    print(f"  Windows: {len(X)} × shape {X.shape[1:]} "
          f"(window={window_size}, step={step}, overlap={100*(1-step/window_size):.0f}%)")
    if y is not None:
        n_abn = y.sum()
        print(f"  Labels: {n_abn} abnormal ({100*n_abn/len(y):.1f}%), "
              f"{len(y)-n_abn} normal")
    return X, y, meta


def extract_stat_features(X, channel_names):
    """
    Ekstrak fitur statistik dari setiap window sebagai alternatif/komplemen
    untuk model yang tidak membutuhkan raw time-series (mis. Isolation Forest).

    Features per channel: mean, std, min, max, peak-to-peak, RMS, skewness, kurtosis
    """
    n_windows, win_size, n_ch = X.shape
    records = []

    for i in range(n_windows):
        row = {}
        for j, ch in enumerate(channel_names):
            alias = CHANNEL_ALIAS.get(ch, ch.replace('FB_',''))
            s = X[i, :, j]
            s_clean = s[~np.isnan(s)]
            if len(s_clean) == 0:
                row.update({f'{alias}_mean': np.nan, f'{alias}_std': np.nan,
                             f'{alias}_min': np.nan,  f'{alias}_max': np.nan,
                             f'{alias}_ptp': np.nan,  f'{alias}_rms': np.nan})
                continue
            row[f'{alias}_mean'] = s_clean.mean()
            row[f'{alias}_std']  = s_clean.std()
            row[f'{alias}_min']  = s_clean.min()
            row[f'{alias}_max']  = s_clean.max()
            row[f'{alias}_ptp']  = s_clean.max() - s_clean.min()
            row[f'{alias}_rms']  = np.sqrt((s_clean**2).mean())
        records.append(row)

    feat_df = pd.DataFrame(records)
    feat_df.to_csv(OUTPUT_DIR / 'p2_stat_features.csv', index=False)
    print(f"  Stat features: {feat_df.shape[0]} windows × {feat_df.shape[1]} features")
    return feat_df


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def run_phase2(df_sel, train_ratio=0.8):
    """
    Jalankan Phase 2 lengkap.

    Args:
        df_sel      : output dari Phase 1 (extract_selected)
        train_ratio : proporsi data normal untuk fit normalizer

    Returns:
        X_train, X_test : windowed arrays
        y_train, y_test : label arrays (jika ada kolom _is_abnormal)
        normalizer      : fitted AdaptiveNormalizer
        corr            : correlation matrix
        feat_train      : stat features train
    """
    print("\n" + "="*65)
    print("  SHMS ANOMALY DETECTION — PHASE 2: PREPROCESSING")
    print("="*65)

    # Pisah normal vs abnormal untuk fitting normalizer
    has_label = '_is_abnormal' in df_sel.columns
    if has_label:
        df_normal = df_sel[~df_sel['_is_abnormal']].copy()
        print(f"  Data normal: {len(df_normal):,} rows")
        print(f"  Data abnormal: {len(df_sel[df_sel['_is_abnormal']]):,} rows")
    else:
        df_normal = df_sel.copy()

    # Channel utama (bukan kontekstual) untuk model
    main_cols = [c for c in ALL_CHANNELS
                 if c in df_sel.columns and c not in CONTEXT_CHANNELS]
    ctx_cols  = [c for c in CONTEXT_CHANNELS if c in df_sel.columns]
    print(f"\n  Channel utama (model input): {len(main_cols)}")
    print(f"  Channel kontekstual       : {len(ctx_cols)}")

    # 2A. Fit normalizer pada data normal saja
    print("\n[2A] Adaptive normalization...")
    norm = AdaptiveNormalizer()
    norm.fit(df_normal, cols=main_cols + ctx_cols)
    df_norm = norm.fit_transform(df_sel, cols=main_cols + ctx_cols)
    norm.save(OUTPUT_DIR / 'p2_normalizer_stats.csv')

    # 2B. Correlation matrix (dari data normal, post-normalisasi)
    print("\n[2B] Correlation matrix...")
    df_norm_normal = norm.transform(df_normal)
    corr  = compute_correlation_matrix(df_norm_normal, cols=main_cols)
    edges = build_graph_edges(corr, threshold=0.5)

    # 2C. Sliding window
    print("\n[2C] Sliding window segmentation...")
    label_col = '_is_abnormal' if has_label else None
    X_all, y_all, meta_all = sliding_window(
        df_norm, cols=main_cols,
        window_size=WINDOW_SIZE, step=WINDOW_STEP,
        label_col=label_col
    )

    # Train/test split — urutan waktu (bukan random!)
    split = int(len(X_all) * train_ratio)
    X_train, X_test = X_all[:split], X_all[split:]
    y_train = y_all[:split] if y_all is not None else None
    y_test  = y_all[split:] if y_all is not None else None

    print(f"\n  Train windows: {len(X_train)}")
    print(f"  Test windows : {len(X_test)}")

    # 2D. Statistical features (untuk Isolation Forest)
    print("\n[2D] Extracting statistical features...")
    feat_train = extract_stat_features(X_train, main_cols)
    feat_test  = extract_stat_features(X_test,  main_cols)
    feat_test.to_csv(OUTPUT_DIR / 'p2_stat_features_test.csv', index=False)

    # Simpan arrays
    np.save(OUTPUT_DIR / 'p2_X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'p2_X_test.npy',  X_test)
    if y_train is not None:
        np.save(OUTPUT_DIR / 'p2_y_train.npy', y_train)
        np.save(OUTPUT_DIR / 'p2_y_test.npy',  y_test)

    print("\n" + "="*65)
    print("  PHASE 2 SELESAI → lanjut Phase 3: Model Training")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    print("="*65)

    return X_train, X_test, y_train, y_test, norm, corr, feat_train, edges


if __name__ == '__main__':
    from shms_phase1_loader import quicktest
    df_sel, _ = quicktest('data/raw/SAMPLE_20260329234000.csv')
    results   = run_phase2(df_sel)
