"""
Figure 1. Aβ burden and plasma p-tau217 by age, clinical status and APOE
haplotype.

The script loads DIMI_DATASET_07-09-2026.csv and p-tau217.csv and keeps
participants aged 50 years or older with APOE 2/3, 3/3, 3/4 or 4/4 and a
diagnosis of cognitively unimpaired (CU), mild cognitive impairment (MCI) or
dementia. Participants with co-pathology (COPAT = 1) in
DIMI_DATASET_07-09-2026.csv are excluded, and p-tau217 values must lie within
365 days of the PET scan. The earliest eligible visit is kept per participant
(Overall column) and per participant and diagnosis (CU, MCI and Dementia
columns). Panel a shows Aβ burden (Centiloid) and plasma p-tau217 z-scores
against age, with LOESS curves and 95% bootstrap confidence bands (1,000
resamples). Panel b shows the percentage of Aβ-positive (Centiloid ≥ 12) and
p-tau217-positive participants in 5-year age bins (bins with at least 5
participants); Aβ positivity combines both input files with one record per
participant. Markers indicate age bins in which at least one haplotype differs
from APOE4/4 (omnibus test and pairwise Fisher exact tests, Holm-corrected).

Inputs:  DIMI_DATASET_07-09-2026.csv, p-tau217.csv
Outputs: FIG_1.png and CSV tables with sample selection, overlap between the
         two files, age-bin counts and percentages, test results, LOESS
         estimates and descriptive statistics.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact, chi2_contingency, random_table

warnings.filterwarnings('default')
RNG = np.random.default_rng(42)
RNG_GLOBAL = np.random.default_rng(2026)

N_BOOT = 1000

FONT_SIZE = {
    'panel_label': 16, 'title': 14, 'axis_label': 14,
    'tick': 12, 'legend': 14, 'n_annot': 9,
}
TITLE_BAND_PTS = 25

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Nimbus Sans'],
    'axes.titleweight': 'regular',
    'font.size':        FONT_SIZE['tick'],
    'axes.labelsize':   FONT_SIZE['axis_label'],
    'xtick.labelsize':  FONT_SIZE['tick'],
    'ytick.labelsize':  FONT_SIZE['tick'],
})

palette_apoe    = {
    '44': '#C7353E',
    '34': '#E0B04A',
    '33': '#389F42',
    '23': '#2E86C1',
}
apoe_order      = ['23', '33', '34','44']
APOE_ZBASE = {'44': 40, '34': 30, '33': 20, '23': 10}
apoe_labels     = {
    '44': r'$\it{APOE}$4/4',
    '34': r'$\it{APOE}$3/4',
    '33': r'$\it{APOE}$3/3',
    '23': r'$\it{APOE}$2/3',
}
order_dx        = ['CU', 'MCI', 'Dementia']
order_dx_panels = ['Overall', 'CU', 'MCI', 'Dementia']
OVERALL_LABEL   = 'Overall'
ID_COL          = 'BIDS_SUBJECT'
CL_THRESHOLD    = 12
MAX_DAYS_PTAU_PET = 365
MIN_AGE         = 50

HAP_REF_DOT   = '44'
HAP_OTHERS    = ['34', '33', '23']
MIN_HAP_N     = 5
DOT_ALPHA     = 0.05
GLOBAL_MONTE_CARLO_RESAMPLES = 10000

AGE_BIN_EDGES = [50, 55, 60, 65, 70, 75, 80, 200]
AGE_BIN_LABS  = ['50–<55', '55–<60', '60–<65', '65–<70', '70–<75', '75–<80', '≥80']
AGE_BIN_MIDS  = [52.5,  57.5,  62.5,  67.5,  72.5,  77.5,  82.5]
AGE_X_MIN, AGE_X_MAX = 48.5, 86
MIN_N_BIN     = 5

SOURCE_PRIORITY      = ['DIMI', 'PTAU']
SESSION_TOL_YEARS    = 1.0
KEEP_DISTINCT_VISITS = False

agebin_rows        = []
bin_sig_rows       = []
overlap_audit_rows = []
sample_audit_rows   = []


def add_title_with_box(ax, title, fontsize):
    from matplotlib.patches import Rectangle
    fig    = ax.get_figure()
    ax_h_pts = ax.get_position().height * fig.get_size_inches()[1] * 72
    band_h   = TITLE_BAND_PTS / ax_h_pts
    ax.add_patch(Rectangle(
        xy=(0, 1.0), width=1.0, height=band_h,
        transform=ax.transAxes,
        facecolor='#EBEBEB',
        edgecolor='black',
        linewidth=plt.rcParams.get('axes.linewidth', 0.8),
        clip_on=False,
        zorder=5,
    ))
    ax.text(
        0.5, 1.0 + band_h * 0.5, title,
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=fontsize,
        clip_on=False,
        zorder=6,
    )

def style_ax(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_SIZE['tick'])

def grid_y(ax, yticks):
    ax.set_yticks(yticks)
    ax.yaxis.grid(False)


def loess_bootstrap_ci(x, y, frac=0.75, n_boot=N_BOOT, alpha=0.05, n_grid=150):
    x_grid      = np.linspace(x.min(), x.max(), n_grid)
    sm_orig     = lowess(y, x, frac=frac, return_sorted=True)
    y_grid      = np.interp(x_grid, sm_orig[:, 0], sm_orig[:, 1])
    boot_curves = np.empty((n_boot, n_grid))
    n = len(x)
    for b in range(n_boot):
        idx = RNG.choice(n, n, replace=True)
        sm_ = lowess(y[idx], x[idx], frac=frac, return_sorted=True)
        boot_curves[b] = np.interp(x_grid, sm_[:, 0], sm_[:, 1])
    lo = np.percentile(boot_curves, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot_curves, 100 * (1 - alpha / 2), axis=0)
    return x_grid, y_grid, lo, hi

def plot_loess_row(ax, data, x_col, y_col, frac=0.75, marker='o'):
    for apoe in apoe_order:
        grp = data[data['apoe_key'] == apoe].dropna(subset=[x_col, y_col])
        if len(grp) < 10:
            continue
        color = palette_apoe[apoe]
        zb = APOE_ZBASE[apoe]
        ax.scatter(grp[x_col], grp[y_col], color=color, alpha=0.18, s=16,
                   marker=marker, linewidths=0, zorder=zb)
        try:
            xg, yg, lo, hi = loess_bootstrap_ci(
                grp[x_col].values.astype(float),
                grp[y_col].values.astype(float), frac=frac)
            ax.fill_between(xg, lo, hi, color=color, alpha=0.20, zorder=zb + 1)
            ax.plot(xg, yg, color=color, lw=2, zorder=zb + 2)
        except Exception as e:
            print(f"  [loess error] {apoe}: {e}")


def _draw_freq_line(ax, xs, ys, color, ls, filled, marker='o', zbase=4,s=40):
    seg_x, seg_y = [], []
    for xv, yv in zip(xs, ys):
        if xv is None:
            if len(seg_x) >= 2:
                ax.plot(seg_x, seg_y, color=color, lw=1.5,
                        ls=ls, zorder=zbase, alpha=0.9)
            seg_x, seg_y = [], []
        else:
            seg_x.append(xv)
            seg_y.append(yv)
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=color, lw=1.5,
                ls=ls, zorder=zbase, alpha=0.9)
    dot_x = [xv for xv in xs if xv is not None]
    dot_y = [yv for yv in ys if yv is not None]
    if dot_x:
        ax.scatter(dot_x, dot_y, marker=marker,
                   color=color if filled else 'white',
                   edgecolors=color, s=s, zorder=zbase + 1,
                   linewidths=0.9 if not filled else 0)


def _compute_agebin(data, outcome_col, age_col, apoe,
                    dataset_tag, dx_label):
    grp = data[data['apoe_key'] == apoe].dropna(subset=[outcome_col, age_col])
    xs, ys = [], []
    for lo_e, hi_e, xm, lab in zip(
            AGE_BIN_EDGES[:-1], AGE_BIN_EDGES[1:],
            AGE_BIN_MIDS, AGE_BIN_LABS):
        sub = grp[(grp[age_col] >= lo_e) & (grp[age_col] < hi_e)]
        n   = len(sub)
        if n < MIN_N_BIN:
            xs.append(None)
            ys.append(None)
            agebin_rows.append({
                'dataset': dataset_tag, 'DX': dx_label, 'APOE': apoe,
                'age_bin': lab, 'age_bin_mid': xm,
                'n': n, 'n_positive': 0,
                'pct_positive': np.nan, 'plotted': 0,
                'subject_id_column': ID_COL,
                'max_abs_days_pTau_PET': (MAX_DAYS_PTAU_PET if outcome_col != 'AB_CL' else np.nan),
                'outcome_definition': (f'Aβ positive if CL ≥ {CL_THRESHOLD}' if outcome_col == 'AB_CL' else 'p-tau217 positivity from pos_2sd_youden'),
            })
        else:
            pct = float(sub[outcome_col].sum() / n * 100)
            xs.append(xm)
            ys.append(pct)
            agebin_rows.append({
                'dataset': dataset_tag, 'DX': dx_label, 'APOE': apoe,
                'age_bin': lab, 'age_bin_mid': xm,
                'n': int(n), 'n_positive': int(sub[outcome_col].sum()),
                'pct_positive': pct, 'plotted': 1,
                'subject_id_column': ID_COL,
                'max_abs_days_pTau_PET': (MAX_DAYS_PTAU_PET if outcome_col != 'AB_CL' else np.nan),
                'outcome_definition': (f'Aβ positive if CL ≥ {CL_THRESHOLD}' if outcome_col == 'AB_CL' else 'p-tau217 positivity from pos_2sd_youden'),
            })
    return xs, ys


def _holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


def _global_independence_test(table):
    table = np.asarray(table, dtype=int)
    chi2, asymptotic_p, _, expected = chi2_contingency(table, correction=False)
    expected_min = float(np.min(expected))
    if expected_min >= 5:
        return float(chi2), float(asymptotic_p), expected_min, 'pearson_chi_square'

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    samples = random_table.rvs(
        row_sums,
        col_sums,
        size=GLOBAL_MONTE_CARLO_RESAMPLES,
        random_state=RNG_GLOBAL,
    )
    sampled_stats = (((samples - expected) ** 2) / expected).sum(axis=(1, 2))
    p_mc = (1 + np.count_nonzero(sampled_stats >= chi2 - 1e-12)) / (
        GLOBAL_MONTE_CARLO_RESAMPLES + 1
    )
    return float(chi2), float(p_mc), expected_min, 'fixed_margin_monte_carlo_exact'


def _bin_significance(data, outcome_col, age_col, dataset_tag, dx_label):
    bin_results = []

    for lo_e, hi_e, xm, lab in zip(
            AGE_BIN_EDGES[:-1], AGE_BIN_EDGES[1:],
            AGE_BIN_MIDS, AGE_BIN_LABS):
        bin_df = data[(data[age_col] >= lo_e) & (data[age_col] < hi_e)].copy()
        bin_df = bin_df.dropna(subset=[outcome_col, 'apoe_key'])

        counts = {}
        for hap in apoe_order:
            g = bin_df[bin_df['apoe_key'] == hap]
            vals = pd.to_numeric(g[outcome_col], errors='coerce').dropna()
            pos = int(vals.sum())
            counts[hap] = {'n': int(len(vals)), 'pos': pos, 'neg': int(len(vals) - pos)}

        global_haps = [h for h in apoe_order if counts[h]['n'] >= MIN_HAP_N]
        global_chi2 = np.nan
        global_p_raw = np.nan
        global_expected_min = np.nan
        global_method = ''
        if len(global_haps) >= 2:
            table = np.array(
                [[counts[h]['pos'], counts[h]['neg']] for h in global_haps],
                dtype=int,
            )
            if table[:, 0].sum() > 0 and table[:, 1].sum() > 0:
                try:
                    (global_chi2, global_p_raw,
                     global_expected_min, global_method) = _global_independence_test(table)
                except Exception as exc:
                    global_method = f'error:{exc}'

        pair_records = []
        ref = counts[HAP_REF_DOT]
        for hap in HAP_OTHERS:
            other = counts[hap]
            testable = (ref['n'] >= MIN_HAP_N) and (other['n'] >= MIN_HAP_N)
            odds_ratio = np.nan
            p_value = np.nan
            note = ''
            if testable:
                table = [[other['pos'], other['neg']], [ref['pos'], ref['neg']]]
                try:
                    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
                except Exception as exc:
                    note = f'error:{exc}'
            else:
                note = 'group_n_below_min'

            pair_records.append({
                'dataset': dataset_tag,
                'DX': dx_label,
                'age_bin': lab,
                'age_bin_mid': xm,
                'contrast': f'{hap}_vs_44',
                'outcome': outcome_col,
                'subject_id_column': ID_COL,
                'max_abs_days_pTau_PET': (
                    MAX_DAYS_PTAU_PET if outcome_col != 'AB_CL' else np.nan
                ),
                'outcome_definition': (
                    f'Aβ positive if CL ≥ {CL_THRESHOLD}'
                    if outcome_col == 'AB_CL'
                    else 'p-tau217 positivity from pos_2sd_youden'
                ),
                'n_44': ref['n'],
                'n_other': other['n'],
                'pos_44': ref['pos'],
                'pos_other': other['pos'],
                'odds_ratio': (
                    float(odds_ratio) if np.isfinite(odds_ratio) else np.nan
                ),
                'p_value_raw': (
                    float(p_value) if np.isfinite(p_value) else np.nan
                ),
                'tested': int(testable and np.isfinite(p_value)),
                'note': note,
                'global_tested': int(np.isfinite(global_p_raw)),
                'global_chi2': (
                    float(global_chi2) if np.isfinite(global_chi2) else np.nan
                ),
                'global_p_value_raw': (
                    float(global_p_raw) if np.isfinite(global_p_raw) else np.nan
                ),
                'global_expected_min': global_expected_min,
                'global_test_method': global_method,
                'global_haplotypes': ','.join(global_haps),
                'global_resamples': (
                    GLOBAL_MONTE_CARLO_RESAMPLES
                    if global_method == 'fixed_margin_monte_carlo_exact' else 0
                ),
            })

        bin_results.append({
            'mid': xm,
            'global_p_raw': global_p_raw,
            'pair_records': pair_records,
        })

    global_adj = _holm_adjust([b['global_p_raw'] for b in bin_results])
    all_pair_records = [r for b in bin_results for r in b['pair_records']]
    pair_adj = _holm_adjust([r['p_value_raw'] for r in all_pair_records])

    pair_index = 0
    sig_mids = set()
    for b, p_global_adj in zip(bin_results, global_adj):
        bin_has_sig_pair = False
        for rec in b['pair_records']:
            p_pair_adj = pair_adj[pair_index]
            pair_index += 1
            rec['global_p_value_holm_across_bins'] = (
                float(p_global_adj) if np.isfinite(p_global_adj) else np.nan
            )
            rec['p_value_holm_across_bins_and_contrasts'] = (
                float(p_pair_adj) if np.isfinite(p_pair_adj) else np.nan
            )
            rec['significant_global_holm'] = int(
                np.isfinite(p_global_adj) and p_global_adj < DOT_ALPHA
            )
            rec['significant_pairwise_holm'] = int(
                np.isfinite(p_pair_adj) and p_pair_adj < DOT_ALPHA
            )
            rec['significant'] = int(
                rec['significant_global_holm']
                and rec['significant_pairwise_holm']
            )
            bin_has_sig_pair = bin_has_sig_pair or bool(rec['significant'])
            bin_sig_rows.append(rec)
        if bin_has_sig_pair:
            sig_mids.add(b['mid'])

    return sig_mids


SIG_AB_COLOR  = '#111111'
SIG_PT_COLOR  = '#111111'
SIG_Y         = 107


def plot_positivity_row(ax, data, outcome, age_col, dataset_tag, dx_label,
                        kind):
    is_ab    = (kind == 'AB')
    line_ls  = '-'
    filled   = True
    pt_marker = 'o' if is_ab else '^'
    s=38 if is_ab else 50
    for apoe in apoe_order:
        color = palette_apoe[apoe]
        xs, ys = _compute_agebin(data, outcome, age_col, apoe,
                                 dataset_tag, dx_label)
        _draw_freq_line(ax, xs, ys, color, ls=line_ls, filled=filled,
                        marker=pt_marker, zbase=APOE_ZBASE[apoe],s=s)

    sig = _bin_significance(data, outcome, age_col, dataset_tag, dx_label)
    sig_marker = 'o' if is_ab else '^'
    sig_color  = SIG_AB_COLOR if is_ab else SIG_PT_COLOR
    sig_size   = 30 if is_ab else 36
    for xm in sig:
        ax.scatter([xm], [SIG_Y], marker=sig_marker, color=sig_color,
                   s=sig_size, zorder=60, clip_on=False)

    ax.set_xlim(AGE_X_MIN, AGE_X_MAX)
    ax.set_xticks(AGE_BIN_MIDS)
    ax.set_xticklabels(AGE_BIN_LABS,
                       fontsize=FONT_SIZE['tick'] - 0.5,
                       rotation=30, ha='right')
    ax.set_xlabel('Age (years)', fontsize=FONT_SIZE['axis_label'])
    ax.set_ylim(-5, 115)
    grid_y(ax, [0, 20, 40, 60, 80, 100])
    style_ax(ax)


APOE_ALLOWED = ['23', '33', '34', '44']


def _normalise_apoe(series):
    values = series.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    return values.str.replace('/', '', regex=False).str.replace('ε', '', regex=False)


def _validate_id(df, source_name):
    if ID_COL not in df.columns:
        raise KeyError(f"{source_name}: required subject identifier '{ID_COL}' not found")
    n_missing = int(df[ID_COL].isna().sum())
    if n_missing:
        raise ValueError(f"{source_name}: {n_missing} rows have missing {ID_COL}; refusing unsafe deduplication")


def prep_dimi(df):
    df = df.copy()
    _validate_id(df, 'DIMI_DATASET_07-09-2026.csv')
    for col in ['AGE', 'CL', 'AB STATUS', 'DX', 'COPAT']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['APOE'] = _normalise_apoe(df['APOE'])
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df = df[
        df['APOE'].isin(APOE_ALLOWED)
        & df['DX'].isin([1, 2, 3])
        & ~df['COPAT'].eq(1)
        & df['AGE'].ge(MIN_AGE)
    ].copy()
    df['DX_LABEL'] = df['DX'].map({1: 'CU', 2: 'MCI', 3: 'Dementia'})
    df['apoe_key'] = df['APOE']
    return df


def prep_ptau(df):
    df = df.copy()
    _validate_id(df, 'p-tau217.csv')
    for col in ['AGE_at_visit', 'pos_2sd_youden', 'zscore_2sd', 'CL',
                'days_pTau_to_PET', 'DX']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['APOE'] = _normalise_apoe(df['APOE'])
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df = df[
        df['APOE'].isin(APOE_ALLOWED)
        & df['DX'].isin([1, 2, 3])
        & df['AGE_at_visit'].ge(MIN_AGE)
    ].copy()
    df['DX_LABEL'] = df['DX'].map({1: 'CU', 2: 'MCI', 3: 'Dementia'})
    df['apoe_key'] = df['APOE']
    df['abs_days_pTau_PET'] = df['days_pTau_to_PET'].abs()
    return df


def _validate_binary_column(df, col, source_name):
    values = set(pd.to_numeric(df[col], errors='coerce').dropna().unique())
    invalid = values.difference({0, 1, 0.0, 1.0})
    if invalid:
        raise ValueError(f"{source_name}: {col} contains non-binary values: {sorted(invalid)}")


def _audit_selection(label, raw_n, eligible_df, selected_df, key_cols):
    sample_audit_rows.append({
        'analysis': label,
        'prepared_rows': int(raw_n),
        'eligible_rows_before_dedup': int(len(eligible_df)),
        'selected_rows_after_dedup': int(len(selected_df)),
        'unique_analysis_units': int(selected_df[key_cols].drop_duplicates().shape[0]),
        'deduplication_key': '+'.join(key_cols),
        'max_abs_days_pTau_PET': MAX_DAYS_PTAU_PET if 'abs_days_pTau_PET' in eligible_df.columns else np.nan,
        'subject_id_column': ID_COL,
    })


def _dedup_by_dx(df, age_col):
    return (df.sort_values([ID_COL, 'DX_LABEL', age_col])
              .drop_duplicates([ID_COL, 'DX_LABEL'], keep='first')
              .reset_index(drop=True))


def _dedup_overall(df, age_col):
    return (df.sort_values([ID_COL, age_col])
              .drop_duplicates(ID_COL, keep='first')
              .reset_index(drop=True))


def select_dimi(df, required_cols, overall=False, label='DIMI'):
    eligible = df.dropna(subset=['AGE', *required_cols]).copy()
    selected = _dedup_overall(eligible, 'AGE') if overall else _dedup_by_dx(eligible, 'AGE')
    keys = [ID_COL] if overall else [ID_COL, 'DX_LABEL']
    _audit_selection(label, len(df), eligible, selected, keys)
    return selected


def select_ptau(df, required_cols, overall=False, label='pTau'):
    eligible = df[
        df['abs_days_pTau_PET'].le(MAX_DAYS_PTAU_PET)
    ].dropna(subset=['AGE_at_visit', *required_cols]).copy()
    selected = (_dedup_overall(eligible, 'AGE_at_visit')
                if overall else _dedup_by_dx(eligible, 'AGE_at_visit'))
    keys = [ID_COL] if overall else [ID_COL, 'DX_LABEL']
    _audit_selection(label, len(df), eligible, selected, keys)
    return selected

def _resolve_source_overlap(combined, key_cols, age_col='age', tag=''):
    d = combined.copy()
    rank = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    d['_src_rank'] = d['source'].map(rank).fillna(len(rank)).astype(int)

    apoe_discordant_keys = []
    for key_value, g in d[d.duplicated(key_cols, keep=False)].groupby(key_cols,
                                                              sort=False):
        if g['source'].nunique() < 2:
            continue
        ages = pd.to_numeric(g[age_col], errors='coerce').to_numpy(dtype=float)
        gap  = float(np.nanmax(ages) - np.nanmin(ages))
        rec = {
            'tag': tag, 'unit': '+'.join(key_cols),
            ID_COL: g[ID_COL].iloc[0],
            'DX_LABEL': (g['DX_LABEL'].iloc[0]
                         if 'DX_LABEL' in key_cols else OVERALL_LABEL),
            'n_rows': int(len(g)),
            'sources': ','.join(sorted(g['source'].unique())),
            'age_min': float(np.nanmin(ages)), 'age_max': float(np.nanmax(ages)),
            'age_gap': gap,
            'same_session': int(gap < SESSION_TOL_YEARS),
            'AB_CL_values': ','.join(str(int(v)) for v in g['AB_CL']),
            'AB_CL_concordant': int(g['AB_CL'].nunique() == 1),
            'APOE_values': ','.join(sorted(g['apoe_key'].astype(str).unique())),
            'APOE_concordant': int(g['apoe_key'].nunique() == 1),
            'excluded_due_to_APOE_discordance': int(g['apoe_key'].nunique() != 1),
        }
        if 'CL' in g.columns:
            rec['CL_values'] = ','.join(f'{v:.1f}' for v in
                                        pd.to_numeric(g['CL'], errors='coerce'))
        overlap_audit_rows.append(rec)
        if g['apoe_key'].nunique() != 1:
            apoe_discordant_keys.append(key_value if isinstance(key_value, tuple) else (key_value,))

    if apoe_discordant_keys:
        drop_mask = pd.Series(False, index=d.index)
        for key_tuple in apoe_discordant_keys:
            unit_mask = pd.Series(True, index=d.index)
            for col, value in zip(key_cols, key_tuple):
                unit_mask &= d[col].eq(value)
            drop_mask |= unit_mask
        n_bad = int(drop_mask.sum())
        d = d.loc[~drop_mask].copy()
        print(f"  [overlap:{tag}] excluded {len(apoe_discordant_keys)} analysis units "
              f"({n_bad} rows) with discordant APOE across sources")

    n_before = len(d)

    d = d.sort_values(key_cols + [age_col, '_src_rank']).reset_index(drop=True)
    d['_new']     = (d.groupby(key_cols, sort=False)[age_col]
                      .diff().fillna(np.inf) >= SESSION_TOL_YEARS).astype(int)
    d['_session'] = d.groupby(key_cols, sort=False)['_new'].cumsum()

    if not KEEP_DISTINCT_VISITS:
        d = d[d['_session'] == d.groupby(key_cols, sort=False)['_session']
                                .transform('min')]

    d = (d.sort_values(key_cols + ['_session', '_src_rank'])
           .drop_duplicates(key_cols + ['_session'], keep='first')
           .drop(columns=['_new', '_session', '_src_rank'])
           .reset_index(drop=True))

    n_dropped = n_before - len(d)
    print(f"  [overlap:{tag}] unit={'+'.join(key_cols)}  "
          f"{n_before:,} pooled rows → {len(d):,} kept  ({n_dropped:,} dropped)")

    if not KEEP_DISTINCT_VISITS:
        assert not d.duplicated(key_cols).any(), \
            f'overlap resolution failed for unit {key_cols}'
    return d


def _make_ab_freq_data(df_dimi, df_ptau_ab_, key_cols, tag):
    shared = [ID_COL, 'apoe_key', 'DX_LABEL', 'AB_CL', 'CL']
    d_dimi = (df_dimi[shared + ['AGE']]
              .rename(columns={'AGE': 'age'}).copy())
    d_dimi['source'] = 'DIMI'
    if len(df_ptau_ab_) == 0:
        return d_dimi
    d_ptau = (df_ptau_ab_[shared + ['AGE_at_visit']]
              .rename(columns={'AGE_at_visit': 'age'}).copy())
    d_ptau['source'] = 'PTAU'
    combined = pd.concat([d_dimi, d_ptau], ignore_index=True)
    return _resolve_source_overlap(combined, key_cols, tag=tag)


print("Loading data...")
df_dimi_raw = pd.read_csv("DIMI_DATASET_07-09-2026.csv")
df_dimi_prepared = prep_dimi(df_dimi_raw)

df_row1 = select_dimi(df_dimi_prepared, ['CL'], overall=False,
                      label='DIMI CL-by-age per-DX')
df_row1_overall = select_dimi(df_dimi_prepared, ['CL'], overall=True,
                              label='DIMI CL-by-age Overall')

df_rows34 = select_dimi(df_dimi_prepared, ['CL'], overall=False,
                        label='DIMI Aβ positivity per-DX')
df_rows34['AB_CL'] = (df_rows34['CL'] >= CL_THRESHOLD).astype(int)
df_rows34_overall = select_dimi(df_dimi_prepared, ['CL'], overall=True,
                                label='DIMI Aβ positivity Overall')
df_rows34_overall['AB_CL'] = (df_rows34_overall['CL'] >= CL_THRESHOLD).astype(int)

PTAU_AVAILABLE = os.path.exists("p-tau217.csv")
if PTAU_AVAILABLE:
    df_ptau_raw = pd.read_csv("p-tau217.csv")
    df_ptau_prepared = prep_ptau(df_ptau_raw)

    df_ptau_34 = select_ptau(df_ptau_prepared, ['pos_2sd_youden'], overall=False,
                             label='pTau positivity per-DX')
    df_ptau_34_overall = select_ptau(df_ptau_prepared, ['pos_2sd_youden'], overall=True,
                                     label='pTau positivity Overall')
    _validate_binary_column(df_ptau_34, 'pos_2sd_youden', 'p-tau217.csv per-DX')
    _validate_binary_column(df_ptau_34_overall, 'pos_2sd_youden', 'p-tau217.csv Overall')

    df_row2 = select_ptau(df_ptau_prepared, ['zscore_2sd'], overall=False,
                          label='pTau z-score-by-age per-DX')
    df_row2_overall = select_ptau(df_ptau_prepared, ['zscore_2sd'], overall=True,
                                  label='pTau z-score-by-age Overall')

    df_ptau_ab = select_ptau(df_ptau_prepared, ['CL'], overall=False,
                             label='pTau-study Aβ positivity per-DX')
    df_ptau_ab['AB_CL'] = (df_ptau_ab['CL'] >= CL_THRESHOLD).astype(int)
    df_ptau_ab_overall = select_ptau(df_ptau_prepared, ['CL'], overall=True,
                                     label='pTau-study Aβ positivity Overall')
    df_ptau_ab_overall['AB_CL'] = (df_ptau_ab_overall['CL'] >= CL_THRESHOLD).astype(int)
    print(f"p-tau217.csv loaded with |pTau-to-PET| ≤ {MAX_DAYS_PTAU_PET} days.")
else:
    df_ptau_34 = pd.DataFrame()
    df_row2 = pd.DataFrame()
    df_ptau_34_overall = pd.DataFrame()
    df_row2_overall = pd.DataFrame()
    df_ptau_ab = pd.DataFrame()
    df_ptau_ab_overall = pd.DataFrame()
    print("p-tau217.csv not found — pTau rows will be blank.")

if PTAU_AVAILABLE:
    ids_dimi = set(df_rows34[ID_COL].astype(str))
    ids_ptau = set(df_ptau_ab[ID_COL].astype(str))
    n_shared = len(ids_dimi & ids_ptau)
    print(f"\n  Subject-ID overlap DIMI ∩ pTau: {n_shared:,} subjects "
          f"({n_shared / max(len(ids_ptau), 1) * 100:.1f}% of the pTau Aβ sample)")
    if n_shared == 0:
        print("  [CHECK] Zero overlap. Confirm the two files share an ID "
              "convention before treating the cohorts as disjoint.")

df_ab_freq         = _make_ab_freq_data(df_rows34, df_ptau_ab,
                                        [ID_COL, 'DX_LABEL'], tag='per_DX')
df_ab_freq_overall = _make_ab_freq_data(df_rows34_overall, df_ptau_ab_overall,
                                        [ID_COL], tag='overall')
n_ptau_ab = len(df_ptau_ab) if PTAU_AVAILABLE else 0
print(f"  AB freq dataset: {len(df_rows34):,} DIMI + {n_ptau_ab:,} pTau-study "
      f"= {len(df_ab_freq):,} subjects total")

print("\n  Haplotype counts (DIMI, after dedup):")
print(df_rows34['apoe_key'].value_counts().to_string())
if PTAU_AVAILABLE:
    print("\n  Haplotype counts (pTau, after dedup):")
    print(df_ptau_34['apoe_key'].value_counts().to_string())


N_COLS  = 4
panel_w = 0.18
gap     = 0.05
total_w = N_COLS * panel_w + (N_COLS - 1) * gap
start_x = (1.0 - total_w) / 2

ROW_H   = {1: 0.135, 2: 0.135, 3: 0.11, 4: 0.11}

GAP_WITHIN_BLOCK  = 0.07
GAP_BETWEEN_BLOCK = 0.12
BOTTOM_MARGIN     = 0.090

row_bottoms = {}
row_bottoms[4] = BOTTOM_MARGIN
row_bottoms[3] = row_bottoms[4] + ROW_H[4] + GAP_WITHIN_BLOCK
row_bottoms[2] = row_bottoms[3] + ROW_H[3] + GAP_BETWEEN_BLOCK
row_bottoms[1] = row_bottoms[2] + ROW_H[2] + GAP_WITHIN_BLOCK

ROW_YLIM   = {
    1: (-25, 150),
    2: (-1, 6),
    3: (-5, 115),
    4: (-5, 115),
}
ROW_YTICKS = {
    1: [0, 50, 100, 150],
    2: [ 0, 2, 4, 6],
    3: [0, 20, 40, 60, 80, 100],
    4: [0, 20, 40, 60, 80, 100],
}
ROW_YLABEL = {
    1: 'Aβ burden (CL)',
    2: 'Plasma p-tau217 (z)',
    3: 'Aβ positivity (%)',
    4: 'Plasma p-tau217 positivity (%)',
}

BIN_ROWS = {3, 4}

CAPTION_A = (r'$\bf{a.}$  A$\mathrm{\beta}$ burden and plasma p-tau217 level by age, '
             r'clinical status, and $\it{APOE}$ haplotype')
CAPTION_B = (r'$\bf{b.}$  A$\mathrm{\beta}$ positivity and plasma p-tau217 positivity by age, '
             r'clinical status, and $\it{APOE}$ haplotype')

TOP_MARGIN = 0.1
top_of_fig = row_bottoms[1] + ROW_H[1]
fig_h = (top_of_fig + TOP_MARGIN) / 0.052
fig   = plt.figure(figsize=(20, fig_h), dpi=300, facecolor='white')

all_axes = {}

for row_idx in [1, 2, 3, 4]:
    for col, dx_label in enumerate(order_dx_panels):
        panel_x = start_x + col * (panel_w + gap)
        ax = fig.add_axes([
            panel_x, row_bottoms[row_idx], panel_w, ROW_H[row_idx],
        ])
        all_axes[(row_idx, col)] = ax
        ax.set_facecolor('white')
        style_ax(ax)
        ax.set_ylim(ROW_YLIM[row_idx])
        grid_y(ax, ROW_YTICKS[row_idx])
        ax.set_ylabel(ROW_YLABEL[row_idx],
                      fontsize=FONT_SIZE['axis_label'], labelpad=8)

        if row_idx in BIN_ROWS:
            ax.set_xlim(AGE_X_MIN, AGE_X_MAX)
            ax.set_xticks(AGE_BIN_MIDS)
            ax.set_xticklabels(AGE_BIN_LABS,
                               fontsize=FONT_SIZE['tick'] - 0.5,
                               rotation=30, ha='right')
            if row_idx == max(BIN_ROWS):
                ax.set_xlabel('Age (years)', fontsize=FONT_SIZE['axis_label'])
        else:
            ax.set_xlim(50, 100)
            ax.set_xticks([50, 60, 70, 80, 90, 100])
            ax.set_xlabel('Age (years)', fontsize=FONT_SIZE['axis_label'])

        add_title_with_box(ax, dx_label, FONT_SIZE['title'])

cap_x = max(0.01, start_x - 0.055)
band_frac_a = (TITLE_BAND_PTS / 72) / fig_h
fig.text(cap_x, row_bottoms[1] + ROW_H[1] + band_frac_a + 0.012,
         CAPTION_A, ha='left', va='bottom',
         fontsize=FONT_SIZE['panel_label'])
fig.text(cap_x, row_bottoms[3] + ROW_H[3] + band_frac_a + 0.012,
         CAPTION_B, ha='left', va='bottom',
         fontsize=FONT_SIZE['panel_label'])


print("\nRow 1: CL × age [LOESS — raw values]...")
for col, dx_label in enumerate(order_dx_panels):
    ax  = all_axes[(1, col)]
    sub = (df_row1_overall if dx_label == OVERALL_LABEL
           else df_row1[df_row1['DX_LABEL'] == dx_label]).copy()
    plot_loess_row(ax, sub, x_col='AGE', y_col='CL')


if PTAU_AVAILABLE:
    print("Row 2: p-tau217 z-score × age [LOESS — raw values]...")
    for col, dx_label in enumerate(order_dx_panels):
        ax  = all_axes[(2, col)]
        sub = (df_row2_overall if dx_label == OVERALL_LABEL
               else df_row2[df_row2['DX_LABEL'] == dx_label]).copy()
        plot_loess_row(ax, sub, x_col='AGE_at_visit', y_col='zscore_2sd',
                       marker='^')


print("Row 3: Aβ age-bin positivity...")
for col, dx_label in enumerate(order_dx_panels):
    ax     = all_axes[(3, col)]
    sub_ab = (df_ab_freq_overall if dx_label == OVERALL_LABEL
              else df_ab_freq[df_ab_freq['DX_LABEL'] == dx_label]).copy()
    plot_positivity_row(ax, sub_ab, 'AB_CL', 'age',
                        'AB_combined', dx_label, kind='AB')


if PTAU_AVAILABLE:
    print("Row 4: p-tau217 age-bin positivity...")
    for col, dx_label in enumerate(order_dx_panels):
        ax     = all_axes[(4, col)]
        sub_pt = (df_ptau_34_overall if dx_label == OVERALL_LABEL
                  else df_ptau_34[df_ptau_34['DX_LABEL'] == dx_label]).copy()
        plot_positivity_row(ax, sub_pt, 'pos_2sd_youden', 'AGE_at_visit',
                            'pTau', dx_label, kind='pTau')


from matplotlib.legend import Legend

LEGEND_GAP_A = 0.04
LEGEND_GAP_B = 0.05

apoe_handles = [
    Line2D([0], [0], color=palette_apoe[a], lw=2.5, ls='-', label=apoe_labels[a])
    for a in apoe_order
]

y_a_legend = row_bottoms[2] - LEGEND_GAP_A
leg_a = Legend(
    fig, apoe_handles, [h.get_label() for h in apoe_handles],
    loc='upper center', bbox_to_anchor=(0.5, y_a_legend),
    bbox_transform=fig.transFigure,
    ncol=len(apoe_handles), frameon=False, fontsize=FONT_SIZE['legend'],
    columnspacing=2.4, handletextpad=0.6,
)
fig.add_artist(leg_a)

shape_handles = [
    Line2D([0], [0], color='black', lw=2, ls='-', marker='o',
           markersize=9, label='Aβ positivity'),
    Line2D([0], [0], color='black', lw=2, ls='-', marker='^',
           markersize=10, label='p-tau217 positivity'),
]
b_handles = apoe_handles + shape_handles
y_b_legend = row_bottoms[4] - LEGEND_GAP_B
leg_b = Legend(
    fig, b_handles, [h.get_label() for h in b_handles],
    loc='upper center', bbox_to_anchor=(0.5, y_b_legend),
    bbox_transform=fig.transFigure,
    ncol=len(b_handles), frameon=False, fontsize=FONT_SIZE['legend'],
    columnspacing=2.0, handletextpad=0.6,
)
fig.add_artist(leg_b)

plt.savefig("FIG_1.png", dpi=300, bbox_inches='tight',
            pad_inches=0.12, facecolor='white')
print("Saved: FIG_1.png")


print("\n--- Quantitative exports (FIG_1) ---")

sample_audit_df = pd.DataFrame(sample_audit_rows)
sample_audit_df.to_csv("FIG_1_sample_selection_audit.csv", index=False)
print(f"Saved: FIG_1_sample_selection_audit.csv  ({len(sample_audit_df)} rows)")

pd.DataFrame(agebin_rows).to_csv("FIG_1_agebin_positivity.csv", index=False)
print(f"Saved: FIG_1_agebin_positivity.csv  ({len(agebin_rows)} rows)")

overlap_df = pd.DataFrame(overlap_audit_rows)
overlap_df.to_csv("FIG_1_overlap_audit.csv", index=False)
if len(overlap_df):
    n_same_sess = int(overlap_df['same_session'].sum())
    n_disc      = int((overlap_df['AB_CL_concordant'] == 0).sum())
    print(f"Saved: FIG_1_overlap_audit.csv  ({len(overlap_df)} collisions; "
          f"{n_same_sess} same-session; {n_disc} with discordant Aβ status "
          f"between sources)")
else:
    print("Saved: FIG_1_overlap_audit.csv  (0 collisions)")

bin_sig_df = pd.DataFrame(bin_sig_rows)
bin_sig_df.to_csv("FIG_1b_bin_significance.csv", index=False)
n_sig = int(bin_sig_df['significant'].sum()) if len(bin_sig_df) else 0
print(f"Saved: FIG_1b_bin_significance.csv  ({len(bin_sig_df)} rows, "
      f"{n_sig} contrasts significant after omnibus and family-wise Holm correction at α={DOT_ALPHA})")


def _summary_block(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan,
                    q1=np.nan, q3=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=int(len(s)),
        mean=float(s.mean()),
        sd=float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        median=float(s.median()),
        q1=float(s.quantile(0.25)),
        q3=float(s.quantile(0.75)),
        min=float(s.min()),
        max=float(s.max()),
    )

AGE_ANCHORS = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)


def _loess_at_anchors(x, y, anchors, frac=0.75, n_boot=N_BOOT,
                      alpha=0.05, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 10:
        return pd.DataFrame(
            columns=['anchor', 'fit', 'ci_lo', 'ci_hi', 'n', 'in_range'])
    rng      = np.random.default_rng(seed)
    sm0      = lowess(y, x, frac=frac, return_sorted=True)
    fit_main = np.interp(anchors, sm0[:, 0], sm0[:, 1])
    boot     = np.empty((n_boot, len(anchors)))
    for b in range(n_boot):
        idx    = rng.choice(len(x), len(x), replace=True)
        sb     = lowess(y[idx], x[idx], frac=frac, return_sorted=True)
        boot[b] = np.interp(anchors, sb[:, 0], sb[:, 1])
    lo       = np.percentile(boot, 100 * alpha / 2,       axis=0)
    hi       = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    in_range = (anchors >= x.min()) & (anchors <= x.max())
    return pd.DataFrame({
        'anchor':   anchors,
        'fit':      np.where(in_range, fit_main, np.nan),
        'ci_lo':    np.where(in_range, lo,       np.nan),
        'ci_hi':    np.where(in_range, hi,       np.nan),
        'n':        int(len(x)),
        'in_range': in_range.astype(int),
    })

row1_rows = []
for dx in order_dx:
    for apoe in apoe_order:
        sub = (df_row1[(df_row1['DX_LABEL'] == dx) &
                       (df_row1['apoe_key'] == apoe)]
               .dropna(subset=['AGE', 'CL']))
        if len(sub) < 10:
            continue
        out = _loess_at_anchors(sub['AGE'].values, sub['CL'].values, AGE_ANCHORS)
        out.insert(0, 'APOE', apoe)
        out.insert(0, 'DX', dx)
        out['subject_id_column'] = ID_COL
        out['AB_measure'] = 'Centiloid'
        row1_rows.append(out)
for apoe in apoe_order:
    sub = df_row1_overall[df_row1_overall['apoe_key'] == apoe].dropna(
        subset=['AGE', 'CL'])
    if len(sub) < 10:
        continue
    out = _loess_at_anchors(sub['AGE'].values, sub['CL'].values, AGE_ANCHORS)
    out.insert(0, 'APOE', apoe)
    out.insert(0, 'DX', OVERALL_LABEL)
    out['subject_id_column'] = ID_COL
    out['AB_measure'] = 'Centiloid'
    row1_rows.append(out)
if row1_rows:
    pd.concat(row1_rows, ignore_index=True).to_csv(
        "FIG_1_loess_row1_CL_by_age.csv", index=False)
    print(f"Saved: FIG_1_loess_row1_CL_by_age.csv  "
          f"({sum(len(r) for r in row1_rows)} rows)")

if PTAU_AVAILABLE:
    row2_rows = []
    for dx in order_dx:
        for apoe in apoe_order:
            sub = (df_row2[(df_row2['DX_LABEL'] == dx) &
                           (df_row2['apoe_key'] == apoe)]
                   .dropna(subset=['AGE_at_visit', 'zscore_2sd']))
            if len(sub) < 10:
                continue
            out = _loess_at_anchors(sub['AGE_at_visit'].values,
                                    sub['zscore_2sd'].values, AGE_ANCHORS)
            out.insert(0, 'APOE', apoe)
            out.insert(0, 'DX', dx)
            out['subject_id_column'] = ID_COL
            out['max_abs_days_pTau_PET'] = MAX_DAYS_PTAU_PET
            row2_rows.append(out)
    for apoe in apoe_order:
        sub = df_row2_overall[
            df_row2_overall['apoe_key'] == apoe
        ].dropna(subset=['AGE_at_visit', 'zscore_2sd'])
        if len(sub) < 10:
            continue
        out = _loess_at_anchors(sub['AGE_at_visit'].values,
                                sub['zscore_2sd'].values, AGE_ANCHORS)
        out.insert(0, 'APOE', apoe)
        out.insert(0, 'DX', OVERALL_LABEL)
        out['subject_id_column'] = ID_COL
        out['max_abs_days_pTau_PET'] = MAX_DAYS_PTAU_PET
        row2_rows.append(out)
    if row2_rows:
        pd.concat(row2_rows, ignore_index=True).to_csv(
            "FIG_1_loess_row2_zscore_by_age.csv", index=False)
        print(f"Saved: FIG_1_loess_row2_zscore_by_age.csv  "
              f"({sum(len(r) for r in row2_rows)} rows)")


desc_rows = []
for dx in order_dx:
    for apoe in apoe_order + ['ALL']:
        sub = (df_rows34[df_rows34['DX_LABEL'] == dx] if apoe == 'ALL'
               else df_rows34[(df_rows34['DX_LABEL'] == dx) &
                              (df_rows34['apoe_key'] == apoe)])
        rec = {'dataset': 'AB_dimi', 'DX': dx, 'APOE': apoe,
               'n_subjects': len(sub), 'subject_id_column': ID_COL,
               'AB_positivity_definition': f'CL ≥ {CL_THRESHOLD}'}
        for col, prefix in [('AGE', 'age'), ('CL', 'CL')]:
            for k, v in _summary_block(sub[col]).items():
                rec[f'{prefix}_{k}'] = v
        ab = pd.to_numeric(sub['AB_CL'], errors='coerce').dropna()
        rec['pct_AB_positive'] = float((ab == 1).mean() * 100) if len(ab) else np.nan
        rec['n_AB_positive']   = int((ab == 1).sum())
        desc_rows.append(rec)
for apoe in apoe_order + ['ALL']:
    sub = (df_rows34_overall if apoe == 'ALL'
           else df_rows34_overall[df_rows34_overall['apoe_key'] == apoe])
    rec = {'dataset': 'AB_dimi', 'DX': OVERALL_LABEL, 'APOE': apoe,
           'n_subjects': len(sub), 'subject_id_column': ID_COL,
           'AB_positivity_definition': f'CL ≥ {CL_THRESHOLD}'}
    for col, prefix in [('AGE', 'age'), ('CL', 'CL')]:
        for k, v in _summary_block(sub[col]).items():
            rec[f'{prefix}_{k}'] = v
    ab = pd.to_numeric(sub['AB_CL'], errors='coerce').dropna()
    rec['pct_AB_positive'] = float((ab == 1).mean() * 100) if len(ab) else np.nan
    rec['n_AB_positive']   = int((ab == 1).sum())
    desc_rows.append(rec)
if PTAU_AVAILABLE:
    for dx in order_dx:
        for apoe in apoe_order + ['ALL']:
            sub = (df_ptau_34[df_ptau_34['DX_LABEL'] == dx] if apoe == 'ALL'
                   else df_ptau_34[(df_ptau_34['DX_LABEL'] == dx) &
                                   (df_ptau_34['apoe_key'] == apoe)])
            rec = {'dataset': 'pTau', 'DX': dx, 'APOE': apoe,
                   'n_subjects': len(sub), 'subject_id_column': ID_COL,
                   'pTau_positivity_definition': 'pos_2sd_youden',
                   'max_abs_days_pTau_PET': MAX_DAYS_PTAU_PET}
            for col, prefix in [('AGE_at_visit', 'age'), ('zscore_2sd', 'zscore')]:
                if col in sub.columns:
                    for k, v in _summary_block(sub[col]).items():
                        rec[f'{prefix}_{k}'] = v
            pt = pd.to_numeric(sub['pos_2sd_youden'], errors='coerce').dropna()
            rec['pct_pTau_positive'] = (float((pt == 1).mean() * 100)
                                        if len(pt) else np.nan)
            rec['n_pTau_positive']   = int((pt == 1).sum())
            desc_rows.append(rec)
    for apoe in apoe_order + ['ALL']:
        sub = (df_ptau_34_overall if apoe == 'ALL'
               else df_ptau_34_overall[df_ptau_34_overall['apoe_key'] == apoe])
        rec = {'dataset': 'pTau', 'DX': OVERALL_LABEL, 'APOE': apoe,
               'n_subjects': len(sub), 'subject_id_column': ID_COL,
               'pTau_positivity_definition': 'pos_2sd_youden',
               'max_abs_days_pTau_PET': MAX_DAYS_PTAU_PET}
        for col, prefix in [('AGE_at_visit', 'age'), ('zscore_2sd', 'zscore')]:
            if col in sub.columns:
                for k, v in _summary_block(sub[col]).items():
                    rec[f'{prefix}_{k}'] = v
        pt = pd.to_numeric(sub['pos_2sd_youden'], errors='coerce').dropna()
        rec['pct_pTau_positive'] = (float((pt == 1).mean() * 100)
                                    if len(pt) else np.nan)
        rec['n_pTau_positive']   = int((pt == 1).sum())
        desc_rows.append(rec)
pd.DataFrame(desc_rows).to_csv("FIG_1_descriptives.csv", index=False)
print(f"Saved: FIG_1_descriptives.csv  ({len(desc_rows)} rows)")

print("Quantitative exports (FIG_1) complete.\n")
