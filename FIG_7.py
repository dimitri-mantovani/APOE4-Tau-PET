"""
Figure 7. Longitudinal change in plasma p-tau217 and tau PET by baseline Aβ
burden, baseline p-tau217, baseline SUVR, clinical status and APOE haplotype.

The script uses APOE 3/3, 3/4 and 4/4 participants and excludes co-pathology
(COPAT = 1). Longitudinal windows start at the first visit with a given
diagnosis and end at the first visit with a different diagnosis (or at the last
visit); windows must span at least 1 year for p-tau217 and 0.5 years for tau
PET. Baseline Centiloid and p-tau217 are matched to each window by nearest age
within 1 year. Section a shows change in plasma p-tau217; sections b and c show
change in meta-ROI SUVR for [18F]flortaucipir and [18F]MK-6240. Points are
annual slopes per window; curves are adjusted annual rates with 95% confidence
bands from linear mixed models fitted on the repeated measurements with a
subject random intercept, adjusted for baseline age and sex (and diagnosis in
the Overall column), with APOE coded as ε4 dose (0/1/2). Four model
specifications are fitted, each producing one figure: (1) linear baseline
biomarker × ε4 dose; (2) quadratic baseline biomarker × ε4 dose; (3) quadratic
biomarker with ε4 dose modifying the linear term only; (4) quadratic biomarker
with ε4 dose modifying the quadratic term only. Columns are Overall, CU, MCI
and MCI + Dementia.

Inputs:  DIMI_DATASET.csv, p-tau217.csv
Outputs: folder FIG_S3_LMM_E4dose with one PNG per model and CSV tables with
         model terms, contrasts, fitted curves, data points, fit summaries and
         data-flow audits.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import norm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore', category=FutureWarning)

TAU_TRACER_SECTIONS = [
    ('b', 'FTP'),
    ('c', 'MK'),
]

TAU_TRACER_CONFIG = {
    'MK': {
        'suvr_col': 'mk_metaroi',
        'output_tag': 'MK',
        'short_name': 'MK-6240',
        'full_name': '[$^{18}$F]MK-6240',
    },
    'FTP': {
        'suvr_col': 'metaroi',
        'output_tag': 'FTP',
        'short_name': 'flortaucipir',
        'full_name': '[$^{18}$F]flortaucipir',
    },
}

PTAU_BL_WINDOW_YEARS = 365.0 / 365.25
CL_BL_WINDOW_YEARS = 365.0 / 365.25


TITLE_BAND_PTS = 25
FONT_SIZE = {
    'panel_label': 20,
    'title': 20,
    'axis_label': 20,
    'tick': 14,
    'legend': 20,
    'annot': 9,
}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.titleweight': 'regular',
    'font.size': FONT_SIZE['tick'],
    'axes.labelsize': FONT_SIZE['axis_label'],
    'xtick.labelsize': FONT_SIZE['tick'],
    'ytick.labelsize': FONT_SIZE['tick'],
})

FIG_W = 24
FIG_H_REF = 18
PANEL_W = 0.19
ROW_H_UNITS = 0.103
COL_GAP = 0.038

INTRA_SECTION_ROW_GAP = 0.06
INTER_SECTION_GAP = 0.135

START_X = 0.08
BOTTOM_MARGIN = 0.125
TOP_MARGIN = 0.075

ROW_LABEL_X_SHIFT = -0.03
ROW_LABEL_TEXT_GAP = 0.023
ROW_LABEL_Y_OFFSET_UNITS = 0.035
LEGEND_GAP_UNITS = 0.078

palette_apoe = {
    '44': '#C7353E',
    '34': '#E0B04A',
    '33': '#389F42',
}
apoe_order = ['44', '34', '33']
apoe_labels = {
    '44': r'$\it{APOE}$4/4',
    '34': r'$\it{APOE}$3/4',
    '33': r'$\it{APOE}$3/3',
}

APOE_E4_DOSE = {'33': 0.0, '34': 1.0, '44': 2.0}

dx_order = ['Overall', 'CU', 'MCI', 'MCI + Dementia']

PTAU_SECTION_LABEL = (
    r'a. Longitudinal changes in plasma p-tau217 level by baseline Aβ burden, '
    r'baseline p-tau217, clinical status, and $\it{APOE}$ haplotype'
)


def tau_section_label(letter, tracer_key):
    cfg = TAU_TRACER_CONFIG[tracer_key]
    return (
        rf'{letter}. Longitudinal changes in {cfg["full_name"]} tau PET burden in '
        rf'Meta-ROI by baseline Aβ burden, baseline p-tau217, baseline SUVR, '
        rf'clinical status, and $\it{{APOE}}$ haplotype'
    )


SECTION_LABELS = {'a': PTAU_SECTION_LABEL}
for _letter, _key in TAU_TRACER_SECTIONS:
    SECTION_LABELS[_letter] = tau_section_label(_letter, _key)

PTAU_ROW_SPECS = [
    {
        'section': 'a',
        'tracer': None,
        'data_type': 'ptau',
        'xcol': 'CL_BL',
        'x_label': 'Baseline Aβ burden (CL)',
        'ycol': 'PTAU_SLOPE',
        'y_label': r'$\Delta$p-tau217/year',
        'x_lim': (-10, 160),
        'x_ticks': [0, 50, 100, 150],
        'y_lim': (-0.5, 1.5),
        'y_ticks': [0, 0.5, 1.0, 1.5],
        'min_n': 5,
    },
    {
        'section': 'a',
        'tracer': None,
        'data_type': 'ptau',
        'xcol': 'PTAU_Z_BL',
        'x_label': 'Baseline p-tau217 z-score',
        'ycol': 'PTAU_SLOPE',
        'y_label': r'$\Delta$p-tau217/year',
        'x_lim': (-1, 8),
        'x_ticks': [0, 2, 4, 6, 8],
        'y_lim': (-0.5, 1.5),
        'y_ticks': [0, 0.5, 1.0, 1.5],
        'min_n': 5,
    },
]


def tau_row_specs(letter, tracer_key):
    common = {
        'section': letter,
        'tracer': tracer_key,
        'data_type': 'taupet',
        'ycol': 'SLOPE',
        'y_label': r'$\Delta$SUVR/year',
        'y_lim': (-0.04, 0.15),
        'y_ticks': [0.0, 0.05, 0.1, 0.15],
        'min_n': 12,
    }
    return [
        {**common,
         'xcol': 'CL_BL',
         'x_label': 'Baseline Aβ burden (CL)',
         'x_lim': (-10, 180),
         'x_ticks': [0, 25,50, 75, 100, 125,150, 175]},
        {**common,
         'xcol': 'PTAU_Z_BL',
         'x_label': 'Baseline p-tau217 (z)',
         'x_lim': (-2, 10),
         'x_ticks': [0, 2, 4, 6, 8, 10]},
        {**common,
         'xcol': 'SUVR_BL',
         'x_label': 'Baseline Meta-ROI SUVR',
         'x_lim': None,
         'x_ticks': None},
    ]


ROW_SPECS = list(PTAU_ROW_SPECS)
for _letter, _key in TAU_TRACER_SECTIONS:
    ROW_SPECS.extend(tau_row_specs(_letter, _key))

SECTION_ORDER = ['a'] + [letter for letter, _ in TAU_TRACER_SECTIONS]
SECTION_ROWS = {}
for _i, _spec in enumerate(ROW_SPECS, start=1):
    SECTION_ROWS.setdefault(_spec['section'], []).append(_i)

N_ROWS = len(ROW_SPECS)
_bottoms_u = {N_ROWS: BOTTOM_MARGIN}
for _ri in range(N_ROWS - 1, 0, -1):
    _same_section = ROW_SPECS[_ri - 1]['section'] == ROW_SPECS[_ri]['section']
    _gap = INTRA_SECTION_ROW_GAP if _same_section else INTER_SECTION_GAP
    _bottoms_u[_ri] = _bottoms_u[_ri + 1] + ROW_H_UNITS + _gap

_total_u = _bottoms_u[1] + ROW_H_UNITS + ROW_LABEL_Y_OFFSET_UNITS + TOP_MARGIN
_scale = 1.0 / _total_u
FIG_H = FIG_H_REF * _total_u
ROW_H = ROW_H_UNITS * _scale
ROW_LABEL_Y_OFFSET = ROW_LABEL_Y_OFFSET_UNITS * _scale
LEGEND_GAP = LEGEND_GAP_UNITS * _scale
ROW_BOTTOMS = {k: v * _scale for k, v in _bottoms_u.items()}

MIN_FOLLOWUP_YEARS_PTAU = 1
MIN_FOLLOWUP_YEARS_TAUPET = 0.5


APPLY_MAIN_COPAT_EXCLUSION_TO_PTAU = True

SUVR_XLIM_BY_TRACER = {}
RATE_YLIM_BY_TRACER = {}

DATA_FLOW_AUDIT_ROWS = []
LMM_WINDOW_AUDIT_ROWS = []
AXIS_CLIPPING_ROWS = []


def add_title_with_box(ax, title, fontsize):
    fig = ax.get_figure()
    ax_h_pts = ax.get_position().height * fig.get_size_inches()[1] * 72
    band_h = TITLE_BAND_PTS / ax_h_pts
    ax.add_patch(Rectangle(
        xy=(0, 1.0), width=1.0, height=band_h,
        transform=ax.transAxes,
        facecolor='#EBEBEB', edgecolor='black',
        linewidth=plt.rcParams.get('axes.linewidth', 0.8),
        clip_on=False, zorder=5,
    ))
    ax.text(0.5, 1.0 + band_h * 0.5, title,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=fontsize, clip_on=False, zorder=6)


def add_section_label(fig, y, text, start_x):
    if ' ' in text:
        prefix, remainder = text.split(' ', 1)
    else:
        prefix, remainder = text, ''
    x_prefix = max(0.005, start_x + ROW_LABEL_X_SHIFT)
    x_text = x_prefix + ROW_LABEL_TEXT_GAP
    fig.text(x_prefix, y, prefix, ha='left', va='bottom',
             fontsize=FONT_SIZE['title'] + 1, fontweight='bold', color='black')
    if remainder:
        fig.text(x_text, y, remainder, ha='left', va='bottom',
                 fontsize=FONT_SIZE['title'] + 1, fontweight='normal', color='black')


def style_ax(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, labelsize=FONT_SIZE['tick'])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)


def normalize_subject_id(series):
    out = series.astype('string').str.strip()
    bad = out.str.lower().isin(['', 'nan', 'none', 'na', 'n/a', '<na>'])
    out = out.mask(bad, pd.NA)
    return out


def normalize_apoe(series):
    out = series.astype('string').str.strip().str.lower()
    out = out.str.replace(r'\.0$', '', regex=True)
    out = out.str.replace(r'[^0-9]', '', regex=True)
    out = out.replace({'43': '34', '32': '23', '42': '24'})
    bad = out.isin(['', 'nan', 'none', 'na', 'n/a', '<na>'])
    return out.mask(bad, pd.NA)


def infer_BIDS_SUBJECT_id_col(df_in):
    for c in ['BIDS_SUBJECT', 'BIDS_BIDS_SUBJECT']:
        if c in df_in.columns:
            return c
    raise KeyError('No BIDS_SUBJECT identifier column found.')


def infer_ptau_BIDS_SUBJECT_col(df_in):
    for c in ['BIDS_SUBJECT', 'BIDS_BIDS_SUBJECT']:
        if c in df_in.columns:
            return c
    raise KeyError('No pTau BIDS_SUBJECT identifier column found.')


def infer_sex_col(df_in):
    for c in ['SEX', 'Sex', 'sex', 'GENDER', 'Gender', 'gender']:
        if c in df_in.columns:
            return c
    return None


def normalize_sex(series, source_name=''):
    raw = series.astype('string')
    txt = raw.str.strip()
    low = txt.str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype='object')

    male_tokens = {'m', 'male', 'masculino', 'man', 'h', 'homem'}
    female_tokens = {'f', 'female', 'feminino', 'woman', 'mulher'}
    missing_tokens = {'', 'nan', 'none', 'na', 'n/a', '<na>'}

    numeric = pd.to_numeric(txt, errors='coerce')
    observed = set(float(v) for v in numeric.dropna().unique())
    if observed and observed.issubset({0.0, 1.0}):
        out[numeric == 1.0] = 'Male'
        out[numeric == 0.0] = 'Female'
    elif observed and observed.issubset({1.0, 2.0}):
        out[numeric == 1.0] = 'Male'
        out[numeric == 2.0] = 'Female'
    elif observed:
        print(
            f'[sex parsing] {source_name or "input"}: numeric sex values '
            f'{sorted(observed)} match neither the 0/1 nor the 1/2 convention. '
            'These rows are treated as missing sex and audited as such.'
        )

    out[low.isin(male_tokens)] = 'Male'
    out[low.isin(female_tokens)] = 'Female'

    unresolved = out.isna() & low.notna() & ~low.isin(missing_tokens)
    if bool(unresolved.any()):
        offending = sorted(set(txt[unresolved].dropna().astype(str)))[:10]
        print(
            f'[sex parsing] {source_name or "input"}: '
            f'{int(unresolved.sum()):,} rows carry an unrecognized sex value '
            f'{offending}; they are set to missing rather than becoming an extra '
            'C(SEX_MODEL) level.'
        )
    return out


def record_flow(source, stage, frame, subject_col, note=''):
    if frame is None:
        n_rows = 0
        n_subjects = 0
    else:
        n_rows = int(len(frame))
        if subject_col in frame.columns:
            s = frame[subject_col].dropna().astype(str)
            n_subjects = int(s.nunique())
        else:
            n_subjects = 0
    DATA_FLOW_AUDIT_ROWS.append({
        'source': source,
        'stage': stage,
        'n_rows': n_rows,
        'n_subjects': n_subjects,
        'note': note,
    })


def fill_subject_constant(frame, subject_col, value_col):
    out = frame.copy()
    if out.empty or value_col not in out.columns:
        return out, set()

    tmp = out[[subject_col, value_col]].dropna().drop_duplicates()
    if tmp.empty:
        return out, set()
    nuniq = tmp.groupby(subject_col)[value_col].nunique(dropna=True)
    conflicts = set(nuniq[nuniq > 1].index.astype(str))
    valid = nuniq[nuniq == 1].index
    lookup = (
        tmp[tmp[subject_col].isin(valid)]
        .drop_duplicates(subset=[subject_col])
        .set_index(subject_col)[value_col]
    )
    missing = out[value_col].isna() & out[subject_col].notna()
    out.loc[missing, value_col] = out.loc[missing, subject_col].map(lookup)
    return out, conflicts


def build_subject_lookup(frame, subject_col, source_name, copat_col=None):
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=[
            'BIDS_SUBJECT', f'{source_name}_N_ROWS', f'{source_name}_APOE',
            f'{source_name}_APOE_CONFLICT', f'{source_name}_SEX',
            f'{source_name}_SEX_CONFLICT', f'{source_name}_COPAT_ANY1',
        ])

    for subj, g in frame.dropna(subset=[subject_col]).groupby(subject_col, sort=False):
        ap = [str(v) for v in g['APOE'].dropna().unique()] if 'APOE' in g.columns else []
        sx = [str(v) for v in g['SEX_MODEL'].dropna().unique()] if 'SEX_MODEL' in g.columns else []
        copat_any1 = False
        if copat_col and copat_col in g.columns:
            cv = pd.to_numeric(g[copat_col], errors='coerce')
            copat_any1 = bool((cv == 1).any())
        rows.append({
            'BIDS_SUBJECT': str(subj),
            f'{source_name}_N_ROWS': int(len(g)),
            f'{source_name}_APOE': ap[0] if len(ap) == 1 else '',
            f'{source_name}_APOE_CONFLICT': int(len(ap) > 1),
            f'{source_name}_SEX': sx[0] if len(sx) == 1 else '',
            f'{source_name}_SEX_CONFLICT': int(len(sx) > 1),
            f'{source_name}_COPAT_ANY1': int(copat_any1),
        })
    return pd.DataFrame(rows)


def fill_from_subject_lookup(frame, subject_col, lookup, value_col, lookup_col):
    if frame.empty or lookup.empty or lookup_col not in lookup.columns:
        return frame
    out = frame.copy()
    mp = lookup.set_index('BIDS_SUBJECT')[lookup_col]
    missing = out[value_col].isna() & out[subject_col].notna()
    out.loc[missing, value_col] = out.loc[missing, subject_col].astype(str).map(mp)
    return out


def prepare_main_data(path='DIMI_DATASET.csv'):
    raw = pd.read_csv(path)
    raw['_SOURCE_ROW_ID'] = np.arange(len(raw), dtype=int)
    subject_col = infer_BIDS_SUBJECT_id_col(raw)
    raw[subject_col] = normalize_subject_id(raw[subject_col])
    record_flow('DIMI_DATASET', 'raw', raw, subject_col)

    raw['APOE'] = normalize_apoe(raw['APOE'])
    sex_col = infer_sex_col(raw)
    if sex_col is None:
        raise KeyError(
            'Age and sex are required covariates for all LMMs, but no sex column '
            "was found in DIMI_DATASET.csv (tried SEX/Sex/sex/GENDER/Gender/gender)."
        )
    raw['SEX_MODEL'] = normalize_sex(raw[sex_col], source_name='DIMI_DATASET')

    for col in ['AGE', 'CL', 'DX', 'COPAT', 'metaroi', 'mk_metaroi']:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
    if 'COPAT' not in raw.columns:
        raise KeyError('DIMI_DATASET.csv must contain COPAT because COPAT=1 is an analysis exclusion.')

    raw, apoe_conflicts = fill_subject_constant(raw, subject_col, 'APOE')
    raw, sex_conflicts = fill_subject_constant(raw, subject_col, 'SEX_MODEL')
    main_lookup = build_subject_lookup(raw, subject_col, 'MAIN', copat_col='COPAT')
    if apoe_conflicts:
        raw.loc[raw[subject_col].astype(str).isin(apoe_conflicts), 'APOE'] = pd.NA
    if sex_conflicts:
        raw.loc[raw[subject_col].astype(str).isin(sex_conflicts), 'SEX_MODEL'] = pd.NA

    d = raw[raw[subject_col].notna()].copy()
    record_flow('DIMI_DATASET', 'valid_subject_id', d, subject_col,
                note=f'missing subject IDs removed={len(raw)-len(d)}')

    before = d.copy()
    d = d[d['APOE'].isin(['33', '34', '44'])].copy()
    record_flow('DIMI_DATASET', 'valid_APOE_33_34_44', d, subject_col,
                note=f'rows excluded for APOE missing/other={len(before)-len(d)}; '
                     f'subjects with conflicting APOE in raw={len(apoe_conflicts)}')

    before = d.copy()
    d = d[d['COPAT'].fillna(0) != 1].copy()
    record_flow('DIMI_DATASET', 'COPAT_not_1', d, subject_col,
                note=f'rows excluded because COPAT=1={len(before)-len(d)}')

    d['DX_LABEL'] = d['DX'].map({1: 'CU', 2: 'MCI', 3: 'Dementia'})
    d['DX_GROUP'] = d['DX_LABEL'].map(
        lambda x: 'MCI + Dementia' if x in ['MCI', 'Dementia'] else x
    )
    before = d.copy()
    d = d.dropna(subset=['AGE', 'DX_LABEL']).copy()
    record_flow('DIMI_DATASET', 'valid_age_and_DX', d, subject_col,
                note=f'rows excluded for missing AGE or invalid/missing DX={len(before)-len(d)}; '
                     f'subjects with conflicting sex in raw={len(sex_conflicts)}')

    return d, subject_col, main_lookup


def prepare_ptau_data(path='p-tau217.csv', main_lookup=None):
    if not os.path.exists(path):
        DATA_FLOW_AUDIT_ROWS.append({
            'source': 'p-tau217', 'stage': 'file_missing', 'n_rows': 0,
            'n_subjects': 0, 'note': f'{path} not found',
        })
        return pd.DataFrame(), None, pd.DataFrame()

    raw = pd.read_csv(path)
    raw['_SOURCE_ROW_ID'] = np.arange(len(raw), dtype=int)
    subject_col = infer_ptau_BIDS_SUBJECT_col(raw)
    raw[subject_col] = normalize_subject_id(raw[subject_col])
    record_flow('p-tau217', 'raw', raw, subject_col)

    raw['APOE'] = normalize_apoe(raw['APOE'])
    sex_col = infer_sex_col(raw)
    raw['SEX_MODEL'] = (
        normalize_sex(raw[sex_col], source_name='p-tau217')
        if sex_col is not None
        else pd.NA
    )
    for col in ['AGE_at_visit', 'zscore_2sd', 'CL', 'DX']:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

    raw, apoe_conflicts = fill_subject_constant(raw, subject_col, 'APOE')
    raw, sex_conflicts = fill_subject_constant(raw, subject_col, 'SEX_MODEL')

    if main_lookup is not None and not main_lookup.empty:
        raw = fill_from_subject_lookup(raw, subject_col, main_lookup, 'APOE', 'MAIN_APOE')
        raw = fill_from_subject_lookup(raw, subject_col, main_lookup, 'SEX_MODEL', 'MAIN_SEX')

    pt_lookup = build_subject_lookup(raw, subject_col, 'PTAU')
    if apoe_conflicts:
        raw.loc[raw[subject_col].astype(str).isin(apoe_conflicts), 'APOE'] = pd.NA
    if sex_conflicts:
        raw.loc[raw[subject_col].astype(str).isin(sex_conflicts), 'SEX_MODEL'] = pd.NA

    d = raw[raw[subject_col].notna()].copy()
    record_flow('p-tau217', 'valid_subject_id', d, subject_col,
                note=f'missing subject IDs removed={len(raw)-len(d)}')

    before = d.copy()
    d = d[d['APOE'].isin(['33', '34', '44'])].copy()
    record_flow('p-tau217', 'valid_APOE_33_34_44', d, subject_col,
                note=f'rows excluded for APOE missing/other={len(before)-len(d)}; '
                     f'subjects with conflicting APOE in pTau raw={len(apoe_conflicts)}')

    if APPLY_MAIN_COPAT_EXCLUSION_TO_PTAU and main_lookup is not None and not main_lookup.empty:
        copat_map = main_lookup.set_index('BIDS_SUBJECT')['MAIN_COPAT_ANY1']
        d['_MAIN_COPAT_ANY1'] = d[subject_col].astype(str).map(copat_map).fillna(0).astype(int)
        before = d.copy()
        d = d[d['_MAIN_COPAT_ANY1'] != 1].copy()
        record_flow('p-tau217', 'COPAT_not_1_via_DIMI_lookup', d, subject_col,
                    note=f'rows excluded because DIMI subject had COPAT=1={len(before)-len(d)}')
        d = d.drop(columns=['_MAIN_COPAT_ANY1'])

    d['DX_LABEL'] = d['DX'].map({1: 'CU', 2: 'MCI', 3: 'Dementia'})
    d['DX_GROUP'] = d['DX_LABEL'].map(
        lambda x: 'MCI + Dementia' if x in ['MCI', 'Dementia'] else x
    )
    record_flow('p-tau217', 'prepared_no_age_DX_sex_drop_yet', d, subject_col,
                note=f'subjects with conflicting sex in pTau raw={len(sex_conflicts)}')
    return d, subject_col, pt_lookup


def build_cross_sheet_subject_audit(main_lookup, ptau_lookup):
    if main_lookup is None:
        main_lookup = pd.DataFrame()
    if ptau_lookup is None:
        ptau_lookup = pd.DataFrame()
    subjects = sorted(set(main_lookup.get('BIDS_SUBJECT', pd.Series(dtype=str)).astype(str)) |
                      set(ptau_lookup.get('BIDS_SUBJECT', pd.Series(dtype=str)).astype(str)))
    base = pd.DataFrame({'BIDS_SUBJECT': subjects})
    if not main_lookup.empty:
        base = base.merge(main_lookup, on='BIDS_SUBJECT', how='left', validate='one_to_one')
    if not ptau_lookup.empty:
        base = base.merge(ptau_lookup, on='BIDS_SUBJECT', how='left', validate='one_to_one')
    base['IN_MAIN'] = base.get('MAIN_N_ROWS', pd.Series(index=base.index, dtype=float)).notna().astype(int)
    base['IN_PTAU'] = base.get('PTAU_N_ROWS', pd.Series(index=base.index, dtype=float)).notna().astype(int)

    ma = base.get('MAIN_APOE', pd.Series('', index=base.index)).fillna('').astype(str)
    pa = base.get('PTAU_APOE', pd.Series('', index=base.index)).fillna('').astype(str)
    ms = base.get('MAIN_SEX', pd.Series('', index=base.index)).fillna('').astype(str)
    ps = base.get('PTAU_SEX', pd.Series('', index=base.index)).fillna('').astype(str)
    base['APOE_AGREEMENT_WHEN_BOTH_PRESENT'] = np.where(
        (ma != '') & (pa != ''), (ma == pa).astype(int), np.nan
    )
    base['SEX_AGREEMENT_WHEN_BOTH_PRESENT'] = np.where(
        (ms != '') & (ps != ''), (ms == ps).astype(int), np.nan
    )

    main_case = {}
    if not main_lookup.empty:
        for sid in main_lookup['BIDS_SUBJECT'].astype(str):
            main_case.setdefault(sid.casefold(), []).append(sid)
    pt_case = {}
    if not ptau_lookup.empty:
        for sid in ptau_lookup['BIDS_SUBJECT'].astype(str):
            pt_case.setdefault(sid.casefold(), []).append(sid)
    case_only = []
    for sid, im, ip in zip(base['BIDS_SUBJECT'].astype(str), base['IN_MAIN'], base['IN_PTAU']):
        key = sid.casefold()
        counterpart = (pt_case if im and not ip else main_case if ip and not im else {}).get(key, [])
        case_only.append(int(any(x != sid for x in counterpart)))
    base['POTENTIAL_CASE_ONLY_ID_MISMATCH'] = case_only
    return base


def nearest_age_match(candidates, target_age, age_col, target_dx=None,
                      window_years=PTAU_BL_WINDOW_YEARS):
    if candidates.empty or not np.isfinite(target_age):
        return None
    cand = candidates.copy()
    cand[age_col] = pd.to_numeric(cand[age_col], errors='coerce')
    cand = cand.dropna(subset=[age_col])
    if cand.empty:
        return None
    cand['_signed_dt'] = cand[age_col] - float(target_age)
    cand['_abs_dt'] = cand['_signed_dt'].abs()
    cand = cand[cand['_abs_dt'] <= window_years].copy()
    if cand.empty:
        return None
    cand['_same_dx_rank'] = 1
    if target_dx is not None and 'DX_LABEL' in cand.columns:
        cand['_same_dx_rank'] = (cand['DX_LABEL'] != target_dx).astype(int)
    cand['_after_rank'] = (cand['_signed_dt'] > 0).astype(int)
    cand['_source_order'] = np.arange(len(cand))
    cand = cand.sort_values(
        ['_abs_dt', '_same_dx_rank', '_after_rank', '_source_order'],
        kind='mergesort',
    )
    return cand.iloc[0]


DX_STAGE_RANK = {'CU': 0, 'MCI': 1, 'Dementia': 2}


def diagnosis_anchored_windows(grp, age_col, dx_col='DX_LABEL'):
    g = grp.sort_values(age_col).reset_index(drop=True).copy()
    n = len(g)
    if n < 2:
        return

    start = 0
    window_index = 0
    while start < n - 1:
        baseline_dx = g.loc[start, dx_col]

        last_same = start
        while last_same + 1 < n and g.loc[last_same + 1, dx_col] == baseline_dx:
            last_same += 1

        converted = last_same + 1 < n
        end = last_same + 1 if converted else last_same
        w = g.iloc[start:end + 1].copy()
        window_index += 1

        end_dx = w.iloc[-1][dx_col]
        if converted:
            r0 = DX_STAGE_RANK.get(baseline_dx)
            r1 = DX_STAGE_RANK.get(end_dx)
            if r0 is None or r1 is None:
                transition_direction = 'other'
            elif r1 > r0:
                transition_direction = 'progression'
            elif r1 < r0:
                transition_direction = 'regression'
            else:
                transition_direction = 'stable'
        else:
            transition_direction = 'stable'

        yield {
            'window_index': window_index,
            'window': w,
            'baseline_dx': baseline_dx,
            'endpoint_dx': end_dx,
            'converted': bool(converted),
            'transition': f'{baseline_dx}->{end_dx}' if converted else f'{baseline_dx}->same',
            'transition_direction': transition_direction,
            'n_visits_baseline_dx': int((w[dx_col] == baseline_dx).sum()),
        }

        if not converted:
            break

        start = end


def compute_taupet_subject_slopes(data, subject_col, region_col,
                                  min_followup=MIN_FOLLOWUP_YEARS_TAUPET):
    cols = [subject_col, region_col, 'AGE', 'DX_LABEL', 'DX_GROUP', 'APOE', 'SEX_MODEL']
    optional = ['CL', 'BIDS_SESSION', 'SESSION', 'STUDY']
    work = data[cols + [c for c in optional if c in data.columns]].copy()
    work[region_col] = pd.to_numeric(work[region_col], errors='coerce')
    work = work.dropna(subset=[subject_col, region_col, 'AGE', 'DX_LABEL']).copy()

    rows = []
    for subj, grp in work.groupby(subject_col, sort=False):
        grp = grp.sort_values('AGE').copy()
        apoes = grp['APOE'].dropna().unique()
        if len(apoes) != 1 or len(grp) < 2:
            continue

        n_visits_subject = int(len(grp))
        for win_info in diagnosis_anchored_windows(grp, age_col='AGE'):
            w = win_info['window']
            if len(w) < 2:
                continue

            followup = float(w['AGE'].iloc[-1] - w['AGE'].iloc[0])
            if followup < min_followup:
                continue

            x = w['AGE'].to_numpy(dtype=float)
            y = w[region_col].to_numpy(dtype=float)
            xm = x - x.mean()
            denom = float((xm ** 2).sum())
            if denom < 1e-10:
                continue

            bl = w.iloc[0]
            fu = w.iloc[-1]
            baseline_dx = win_info['baseline_dx']
            baseline_dx_group = 'MCI + Dementia' if baseline_dx in ['MCI', 'Dementia'] else baseline_dx

            rows.append({
                'BIDS_SUBJECT': subj,
                'WINDOW_INDEX': int(win_info['window_index']),
                'APOE': apoes[0],
                'SEX_MODEL': bl.get('SEX_MODEL', np.nan),
                'DX_LABEL': baseline_dx,
                'DX_GROUP': baseline_dx_group,
                'DX_FU_LABEL': win_info['endpoint_dx'],
                'CONVERTED_AT_WINDOW_END': int(win_info['converted']),
                'DX_TRANSITION': win_info['transition'],
                'TRANSITION_DIRECTION': win_info['transition_direction'],
                'AGE_BL': float(bl['AGE']),
                'AGE_FU': float(fu['AGE']),
                'FOLLOWUP_YEARS': followup,
                'SUVR_BL': float(bl[region_col]),
                'SUVR_FU': float(fu[region_col]),
                'SLOPE': float((xm * y).sum() / denom),
                'N_VISITS_WINDOW': int(len(w)),
                'N_VISITS_BASELINE_DX': int(win_info['n_visits_baseline_dx']),
                'N_VISITS_TOTAL': n_visits_subject,
                'TAU_SESSION_BL': bl.get('BIDS_SESSION', bl.get('SESSION', np.nan)),
                'TAU_SESSION_FU': fu.get('BIDS_SESSION', fu.get('SESSION', np.nan)),
            })
    return pd.DataFrame(rows)


def attach_ptau_baseline(slope_df, pt, pt_subject_col,
                         window_years=PTAU_BL_WINDOW_YEARS):
    slope_df = slope_df.copy()
    new_cols = {
        'PTAU_Z_BL': np.nan,
        'PTAU_AGE_BL': np.nan,
        'PTAU_DT_YEARS': np.nan,
        'PTAU_SIGNED_DT_YEARS': np.nan,
        'PTAU_MATCH_DIRECTION': '',
        'PTAU_DX_MATCH': np.nan,
        'PTAU_DX_LABEL': '',
        'PTAU_SESSION_BL': '',
        'PTAU_MATCH_STATUS': 'not_matched',
        'PTAU_MATCH_REASON': '',
    }
    for c, default in new_cols.items():
        slope_df[c] = default
    if pt.empty or slope_df.empty:
        return slope_df

    keep = [pt_subject_col, 'AGE_at_visit', 'zscore_2sd', 'DX_LABEL', 'SESSION', 'APOE']
    pt_match = pt[[c for c in keep if c in pt.columns]].dropna(
        subset=[pt_subject_col, 'AGE_at_visit', 'zscore_2sd']
    ).copy()
    grouped = {k: g for k, g in pt_match.groupby(pt_subject_col, sort=False)}

    for idx, row in slope_df.iterrows():
        cand = grouped.get(row['BIDS_SUBJECT'])
        if cand is None:
            slope_df.at[idx, 'PTAU_MATCH_REASON'] = 'subject_not_found_in_ptau'
            continue
        cand = cand[cand['APOE'] == row['APOE']].copy()
        if cand.empty:
            slope_df.at[idx, 'PTAU_MATCH_REASON'] = 'no_same_APOE_candidate'
            continue
        best = nearest_age_match(
            cand, row['AGE_BL'], 'AGE_at_visit', row.get('DX_LABEL'), window_years
        )
        if best is None:
            slope_df.at[idx, 'PTAU_MATCH_REASON'] = 'no_candidate_within_age_window'
            continue
        signed_dt = float(best['_signed_dt'])
        slope_df.at[idx, 'PTAU_Z_BL'] = float(best['zscore_2sd'])
        slope_df.at[idx, 'PTAU_AGE_BL'] = float(best['AGE_at_visit'])
        slope_df.at[idx, 'PTAU_DT_YEARS'] = float(best['_abs_dt'])
        slope_df.at[idx, 'PTAU_SIGNED_DT_YEARS'] = signed_dt
        slope_df.at[idx, 'PTAU_MATCH_DIRECTION'] = (
            'before_or_same' if signed_dt <= 0 else 'after'
        )
        slope_df.at[idx, 'PTAU_DX_MATCH'] = int(best.get('DX_LABEL') == row.get('DX_LABEL'))
        slope_df.at[idx, 'PTAU_DX_LABEL'] = best.get('DX_LABEL', '')
        slope_df.at[idx, 'PTAU_SESSION_BL'] = best.get('SESSION', '')
        slope_df.at[idx, 'PTAU_MATCH_STATUS'] = 'matched'
        slope_df.at[idx, 'PTAU_MATCH_REASON'] = 'matched'
    return slope_df


PTAU_SLOPE_COLUMNS = [
    'BIDS_SUBJECT', 'WINDOW_INDEX', 'APOE', 'SEX_MODEL', 'DX_LABEL', 'DX_GROUP',
    'DX_FU_LABEL', 'CONVERTED_AT_WINDOW_END', 'DX_TRANSITION',
    'TRANSITION_DIRECTION', 'AGE_BL', 'AGE_FU', 'FOLLOWUP_YEARS',
    'PTAU_AGE_BL', 'PTAU_SLOPE', 'PTAU_Z_BL', 'N_VISITS_WINDOW',
    'N_VISITS_BASELINE_DX', 'N_VISITS_TOTAL', 'PTAU_SESSION_BL',
    'PTAU_SESSION_FU',
]


def compute_ptau_subject_slopes(pt, pt_subject_col,
                                min_followup=MIN_FOLLOWUP_YEARS_PTAU):
    if pt.empty:
        return pd.DataFrame(columns=PTAU_SLOPE_COLUMNS)
    rows = []
    work = pt.dropna(subset=[pt_subject_col, 'AGE_at_visit', 'zscore_2sd', 'DX_LABEL']).copy()
    for subj, grp in work.groupby(pt_subject_col, sort=False):
        grp = grp.sort_values('AGE_at_visit').copy()
        apoes = grp['APOE'].dropna().unique()
        if len(apoes) != 1 or len(grp) < 2:
            continue

        n_visits_subject = int(len(grp))
        for win_info in diagnosis_anchored_windows(grp, age_col='AGE_at_visit'):
            w = win_info['window']
            if len(w) < 2:
                continue

            followup = float(w['AGE_at_visit'].iloc[-1] - w['AGE_at_visit'].iloc[0])
            if followup < min_followup:
                continue

            x = w['AGE_at_visit'].to_numpy(dtype=float)
            y = w['zscore_2sd'].to_numpy(dtype=float)
            xm = x - x.mean()
            denom = float((xm ** 2).sum())
            if denom < 1e-10:
                continue

            bl = w.iloc[0]
            fu = w.iloc[-1]
            baseline_dx = win_info['baseline_dx']
            baseline_dx_group = 'MCI + Dementia' if baseline_dx in ['MCI', 'Dementia'] else baseline_dx

            rows.append({
                'BIDS_SUBJECT': subj,
                'WINDOW_INDEX': int(win_info['window_index']),
                'APOE': apoes[0],
                'SEX_MODEL': bl.get('SEX_MODEL', np.nan),
                'DX_LABEL': baseline_dx,
                'DX_GROUP': baseline_dx_group,
                'DX_FU_LABEL': win_info['endpoint_dx'],
                'CONVERTED_AT_WINDOW_END': int(win_info['converted']),
                'DX_TRANSITION': win_info['transition'],
                'TRANSITION_DIRECTION': win_info['transition_direction'],
                'AGE_BL': float(bl['AGE_at_visit']),
                'AGE_FU': float(fu['AGE_at_visit']),
                'FOLLOWUP_YEARS': followup,
                'PTAU_AGE_BL': float(bl['AGE_at_visit']),
                'PTAU_SLOPE': float((xm * y).sum() / denom),
                'PTAU_Z_BL': float(bl['zscore_2sd']),
                'N_VISITS_WINDOW': int(len(w)),
                'N_VISITS_BASELINE_DX': int(win_info['n_visits_baseline_dx']),
                'N_VISITS_TOTAL': n_visits_subject,
                'PTAU_SESSION_BL': bl.get('BIDS_SESSION', bl.get('SESSION', np.nan)),
                'PTAU_SESSION_FU': fu.get('BIDS_SESSION', fu.get('SESSION', np.nan)),
            })
    return pd.DataFrame(rows, columns=PTAU_SLOPE_COLUMNS)


def attach_cl_baseline(slope_df, main_data, main_subject_col,
                       window_years=CL_BL_WINDOW_YEARS):
    out = slope_df.copy()
    out['CL_BL'] = np.nan
    out['CL_AGE_BL'] = np.nan
    out['CL_DT_YEARS'] = np.nan
    out['CL_SIGNED_DT_YEARS'] = np.nan
    out['CL_MATCH_DIRECTION'] = ''
    out['CL_DX_MATCH'] = np.nan
    out['CL_DX_LABEL'] = ''
    out['CL_SESSION_BL'] = ''
    out['CL_MATCH_STATUS'] = 'not_matched'
    out['CL_MATCH_REASON'] = ''
    if out.empty:
        return out

    cols = [main_subject_col, 'AGE', 'CL', 'DX_LABEL', 'APOE']
    optional = ['BIDS_SESSION', 'SESSION']
    cl_data = main_data[cols + [c for c in optional if c in main_data.columns]].copy()
    cl_data = cl_data.dropna(subset=[main_subject_col, 'AGE', 'CL'])
    grouped = {k: g for k, g in cl_data.groupby(main_subject_col, sort=False)}

    for idx, row in out.iterrows():
        cand = grouped.get(row['BIDS_SUBJECT'])
        if cand is None:
            out.at[idx, 'CL_MATCH_REASON'] = 'subject_not_found_in_DIMI'
            continue
        cand = cand[cand['APOE'] == row['APOE']].copy()
        if cand.empty:
            out.at[idx, 'CL_MATCH_REASON'] = 'no_same_APOE_candidate'
            continue
        best = nearest_age_match(
            cand, row['AGE_BL'], 'AGE', row.get('DX_LABEL'), window_years
        )
        if best is None:
            out.at[idx, 'CL_MATCH_REASON'] = 'no_candidate_within_age_window'
            continue
        signed_dt = float(best['_signed_dt'])
        out.at[idx, 'CL_BL'] = float(best['CL'])
        out.at[idx, 'CL_AGE_BL'] = float(best['AGE'])
        out.at[idx, 'CL_DT_YEARS'] = float(best['_abs_dt'])
        out.at[idx, 'CL_SIGNED_DT_YEARS'] = signed_dt
        out.at[idx, 'CL_MATCH_DIRECTION'] = 'before_or_same' if signed_dt <= 0 else 'after'
        out.at[idx, 'CL_DX_MATCH'] = int(best.get('DX_LABEL') == row.get('DX_LABEL'))
        out.at[idx, 'CL_DX_LABEL'] = best.get('DX_LABEL', '')
        out.at[idx, 'CL_SESSION_BL'] = best.get('BIDS_SESSION', best.get('SESSION', ''))
        out.at[idx, 'CL_MATCH_STATUS'] = 'matched'
        out.at[idx, 'CL_MATCH_REASON'] = 'matched'
    return out


OUTPUT_DIR = 'FIG_S3_LMM_E4dose'

LMM_MIN_SUBJECTS = 12
LMM_MIN_OBSERVATIONS = 24
MIN_GENOTYPE_MODEL_UNITS_FOR_CURVE = 3
MODEL_N_GRID = 200
LMM_REML = True
LMM_MAXITER = 500
LMM_OPTIMIZERS = ['lbfgs', 'bfgs', 'powell']
LMM_DEDUPLICATE_SHARED_BOUNDARIES = True
CURVE_CONTRAST_TOLERANCE = 1e-8
DESIGN_RANK_TOLERANCE = 1e-10

ADJUST_OVERALL_FOR_DX_MAIN_EFFECT = True
ADJUST_MCI_DEMENTIA_FOR_DX_MAIN_EFFECT = False

MODEL_SPECS = [
    {
        'order': 1,
        'key': 'LMM_E4dose_linearX',
        'x_degree': 1,
        'x_e4_interaction': True,
        'quadratic_x_e4_interaction': False,
        'filename': 'FIG_7.png',
        'title': 'MixedLM: linear baseline biomarker × APOE ε4 dose',
        'description': (
            'Raw repeated biomarker values modeled using TIME × baseline '
            'biomarker × linear epsilon4 dose, with baseline age, sex, optional '
            'diagnosis main-effect adjustment, and a subject random intercept. '
            'Curves are adjusted annual rates.'
        ),
    },
    {
        'order': 2,
        'key': 'LMM_E4dose_quadraticX',
        'x_degree': 2,
        'x_e4_interaction': True,
        'quadratic_x_e4_interaction': True,
        'filename': 'FIG_S3_2_LMM_E4dose_quadraticX.png',
        'title': 'MixedLM: quadratic baseline biomarker × APOE ε4 dose',
        'description': (
            'Raw repeated biomarker values modeled using TIME × (baseline '
            'biomarker + baseline biomarker squared) × linear epsilon4 dose, '
            'with baseline age, sex, optional diagnosis main-effect adjustment, '
            'and a subject random intercept. The quadratic term is unconstrained; '
            'curves are adjusted annual rates.'
        ),
    },
    {
        'order': 3,
        'key': 'LMM_E4dose_quadraticX_linearXbyE4only',
        'x_degree': 2,
        'x_e4_interaction': True,
        'quadratic_x_e4_interaction': False,
        'filename': 'FIG_S3_3_LMM_E4dose_quadraticX_linearXbyE4only.png',
        'title': 'MixedLM: quadratic baseline biomarker with linear X × APOE ε4 dose only',
        'description': (
            'Raw repeated biomarker values modeled using TIME × (baseline biomarker '
            '+ baseline biomarker squared) plus TIME × baseline biomarker × linear '
            'epsilon4 dose, with baseline age, sex, optional diagnosis main-effect '
            'adjustment, and a subject random intercept. Epsilon4 dose may modify '
            'the linear X-related annual-rate slope, while the quadratic curvature '
            'is constrained to be common across epsilon4-dose groups.'
        ),
    },
    {
        'order': 4,
        'key': 'LMM_E4dose_quadraticX_curvatureXbyE4only',
        'x_degree': 2,
        'x_e4_interaction': False,
        'quadratic_x_e4_interaction': True,
        'filename': 'FIG_S3_4_LMM_E4dose_quadraticX_curvatureXbyE4only.png',
        'title': 'MixedLM: quadratic baseline biomarker with curvature × APOE ε4 dose only',
        'description': (
            'Raw repeated biomarker values modeled with a common linear X effect and '
            'an epsilon4-dose interaction restricted to centered X-squared, with '
            'baseline age, sex, optional diagnosis main-effect adjustment, and a '
            'subject random intercept. Epsilon4 dose may modify quadratic curvature, '
            'while the linear X-by-epsilon4-dose interaction is intentionally excluded.'
        ),
    },
]


def filter_dx_group(data, dx_group):
    if dx_group == 'Overall':
        return data.copy()
    if dx_group == 'CU':
        return data[data['DX_LABEL'] == 'CU'].copy()
    if dx_group == 'MCI':
        return data[data['DX_LABEL'] == 'MCI'].copy()
    return data[data['DX_GROUP'] == 'MCI + Dementia'].copy()


DEFAULT_SUVR_XLIM = {
    'CU': ((1.0, 1.85), [1.0, 1.25, 1.5, 1.75]),
    'default': ((1.0, 2.6), [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]),
}


def panel_axis_limits(spec, dx_group):
    tracer = spec.get('tracer')
    y_override = RATE_YLIM_BY_TRACER.get(tracer)
    if y_override is not None:
        y_lim, y_ticks = y_override
    else:
        y_lim, y_ticks = spec['y_lim'], spec['y_ticks']

    if spec['xcol'] == 'SUVR_BL':
        override = SUVR_XLIM_BY_TRACER.get(tracer, {})
        key = 'CU' if dx_group == 'CU' else 'default'
        x_lim, x_ticks = override.get(key, DEFAULT_SUVR_XLIM[key])
    else:
        x_lim, x_ticks = spec['x_lim'], spec['x_ticks']
    return x_lim, list(x_ticks), y_lim, list(y_ticks)


def set_panel_axes(ax, spec, dx_group):
    x_lim, x_ticks, y_lim, y_ticks = panel_axis_limits(spec, dx_group)
    ax.axhline(y=0, color='gray', lw=0.8, linestyle='--', alpha=0.5)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.set_ylabel(spec['y_label'], fontsize=FONT_SIZE['axis_label'])
    ax.set_xlabel(spec['x_label'], fontsize=FONT_SIZE['axis_label'])
    ax.set_xlim(x_lim)
    ax.set_xticks(x_ticks)
    style_ax(ax)


def summarize_series(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return {
            'n': 0, 'mean': np.nan, 'sd': np.nan, 'median': np.nan,
            'q1': np.nan, 'q3': np.nan, 'min': np.nan, 'max': np.nan,
        }
    return {
        'n': int(len(s)),
        'mean': float(s.mean()),
        'sd': float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        'median': float(s.median()),
        'q1': float(s.quantile(0.25)),
        'q3': float(s.quantile(0.75)),
        'min': float(s.min()),
        'max': float(s.max()),
    }


def build_repeated_long_data(data, subject_col, age_col, value_col, slope_df, source_tag):
    out_cols = [
        'BIDS_SUBJECT', 'WINDOW_INDEX', 'WINDOW_UID', 'SOURCE_OBS_ID',
        'APOE', 'DX_LABEL', 'DX_GROUP', 'DX_FU_LABEL', 'DX_TRANSITION',
        'TRANSITION_DIRECTION', 'AGE_BL', 'AGE_VISIT', 'TIME', 'Y_OBS',
        'SEX_MODEL', 'IS_WINDOW_BASELINE', 'CL_BL', 'PTAU_Z_BL', 'SUVR_BL',
        'FOLLOWUP_YEARS', 'N_VISITS_WINDOW', 'N_VISITS_TOTAL',
    ]
    if data.empty or slope_df.empty:
        return pd.DataFrame(columns=out_cols)

    meta = slope_df.copy()
    meta['BIDS_SUBJECT'] = meta['BIDS_SUBJECT'].astype(str)
    meta['WINDOW_INDEX'] = pd.to_numeric(meta['WINDOW_INDEX'], errors='coerce')
    meta = meta.dropna(subset=['BIDS_SUBJECT', 'WINDOW_INDEX']).copy()
    meta['WINDOW_INDEX'] = meta['WINDOW_INDEX'].astype(int)
    meta_lookup = {
        (str(r['BIDS_SUBJECT']), int(r['WINDOW_INDEX'])): r
        for _, r in meta.iterrows()
    }

    cols = [subject_col, age_col, value_col, 'DX_LABEL', 'DX_GROUP', 'APOE', 'SEX_MODEL']
    optional = ['_SOURCE_ROW_ID', 'BIDS_SESSION', 'SESSION', 'STUDY']
    work = data[cols + [c for c in optional if c in data.columns]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
    work[age_col] = pd.to_numeric(work[age_col], errors='coerce')
    work = work.dropna(subset=[subject_col, value_col, age_col, 'DX_LABEL']).copy()

    rows = []
    for subj, grp in work.groupby(subject_col, sort=False):
        subj = str(subj)
        grp = grp.sort_values(age_col).copy()
        for win_info in diagnosis_anchored_windows(grp, age_col=age_col):
            wi = int(win_info['window_index'])
            key = (subj, wi)
            if key not in meta_lookup:
                continue
            m = meta_lookup[key]
            w = win_info['window'].copy()
            age_bl = float(m['AGE_BL'])
            window_uid = f'{source_tag}:{subj}:w{wi}'
            for pos, (_, vr) in enumerate(w.iterrows()):
                source_row = vr.get('_SOURCE_ROW_ID', np.nan)
                if pd.notna(source_row):
                    source_obs_id = f'{source_tag}:{subj}:r{int(source_row)}'
                else:
                    source_obs_id = f'{source_tag}:{subj}:age{float(vr[age_col]):.8f}'
                rows.append({
                    'BIDS_SUBJECT': subj,
                    'WINDOW_INDEX': wi,
                    'WINDOW_UID': window_uid,
                    'SOURCE_OBS_ID': source_obs_id,
                    'APOE': m.get('APOE', vr.get('APOE', np.nan)),
                    'DX_LABEL': m.get('DX_LABEL', win_info['baseline_dx']),
                    'DX_GROUP': m.get('DX_GROUP', np.nan),
                    'DX_FU_LABEL': m.get('DX_FU_LABEL', win_info['endpoint_dx']),
                    'DX_TRANSITION': m.get('DX_TRANSITION', win_info['transition']),
                    'TRANSITION_DIRECTION': m.get(
                        'TRANSITION_DIRECTION', win_info['transition_direction']
                    ),
                    'AGE_BL': age_bl,
                    'AGE_VISIT': float(vr[age_col]),
                    'TIME': float(vr[age_col]) - age_bl,
                    'Y_OBS': float(vr[value_col]),
                    'SEX_MODEL': m.get('SEX_MODEL', vr.get('SEX_MODEL', np.nan)),
                    'IS_WINDOW_BASELINE': int(pos == 0),
                    'CL_BL': pd.to_numeric(m.get('CL_BL', np.nan), errors='coerce'),
                    'PTAU_Z_BL': pd.to_numeric(m.get('PTAU_Z_BL', np.nan), errors='coerce'),
                    'SUVR_BL': pd.to_numeric(m.get('SUVR_BL', np.nan), errors='coerce'),
                    'FOLLOWUP_YEARS': pd.to_numeric(
                        m.get('FOLLOWUP_YEARS', np.nan), errors='coerce'
                    ),
                    'N_VISITS_WINDOW': pd.to_numeric(
                        m.get('N_VISITS_WINDOW', np.nan), errors='coerce'
                    ),
                    'N_VISITS_TOTAL': pd.to_numeric(
                        m.get('N_VISITS_TOTAL', np.nan), errors='coerce'
                    ),
                })
    return pd.DataFrame(rows, columns=out_cols)


def _empty_prep():
    return {
        'n_removed_shared_boundary_rows': 0,
        'x_centering_mean': np.nan,
        'baseline_age_centering_mean': np.nan,
        'sex_levels': '',
        'n_units_before_requirements': 0,
        'n_units_after_requirements': 0,
    }


def _clean_sex(series):
    out = series.astype('string').str.strip()
    bad = out.str.lower().isin(['', 'nan', 'none', '<na>', 'na', 'n/a'])
    return out.mask(bad, pd.NA)


def prepare_lmm_panel(long_data, spec, dx_group, model_spec):
    if long_data.empty:
        return pd.DataFrame(), _empty_prep()

    d0 = filter_dx_group(long_data, dx_group).copy()
    if d0.empty:
        return d0, _empty_prep()

    d0['TIME'] = pd.to_numeric(d0['TIME'], errors='coerce')
    d0['Y_OBS'] = pd.to_numeric(d0['Y_OBS'], errors='coerce')
    d0['AGE_BL'] = pd.to_numeric(d0['AGE_BL'], errors='coerce')
    d0['X_RAW'] = pd.to_numeric(d0[spec['xcol']], errors='coerce')
    d0['SEX_MODEL'] = _clean_sex(d0['SEX_MODEL'])
    d0['E4_DOSE'] = d0['APOE'].map(APOE_E4_DOSE)

    n_before = int(d0['WINDOW_UID'].nunique())
    for window_uid, w in d0.groupby('WINDOW_UID', sort=False):
        reasons = []
        has_x = bool(w['X_RAW'].notna().all())
        has_age = bool(w['AGE_BL'].notna().all())
        has_sex = bool(w['SEX_MODEL'].notna().all())
        has_apoe = bool(w['E4_DOSE'].notna().all())
        n_valid = int((w['TIME'].notna() & w['Y_OBS'].notna()).sum())
        if not has_x:
            reasons.append('missing_baseline_predictor')
        if not has_age:
            reasons.append('missing_baseline_age')
        if not has_sex:
            reasons.append('missing_sex')
        if not has_apoe:
            reasons.append('invalid_or_missing_APOE')
        if n_valid < 2:
            reasons.append('fewer_than_two_valid_repeated_measurements')
        LMM_WINDOW_AUDIT_ROWS.append({
            'model_key': model_spec['key'],
            'model_family': 'LMM',
            'x_degree': model_spec['x_degree'],
            'x_e4_interaction': bool(model_spec['x_e4_interaction']),
            'quadratic_x_e4_interaction': bool(model_spec.get('quadratic_x_e4_interaction', False)),
            'section': spec['section'],
            'tracer': spec.get('tracer') or 'PTAU',
            'data_type': spec['data_type'],
            'subplot': dx_group,
            'x_variable': spec['xcol'],
            'WINDOW_UID': window_uid,
            'BIDS_SUBJECT': str(w['BIDS_SUBJECT'].iloc[0]),
            'APOE': str(w['APOE'].iloc[0]),
            'DX_LABEL': w['DX_LABEL'].iloc[0],
            'n_repeated_rows_before_filters': int(len(w)),
            'n_valid_outcome_time_rows': n_valid,
            'has_baseline_predictor': int(has_x),
            'has_baseline_age': int(has_age),
            'has_sex': int(has_sex),
            'has_valid_APOE': int(has_apoe),
            'included_before_shared_boundary_dedup': int(len(reasons) == 0),
            'exclusion_reason': ';'.join(reasons) if reasons else 'included',
        })

    valid = (
        d0['TIME'].notna() & d0['Y_OBS'].notna() & d0['AGE_BL'].notna() &
        d0['SEX_MODEL'].notna() & d0['X_RAW'].notna() & d0['E4_DOSE'].notna()
    )
    d = d0.loc[valid & d0['APOE'].isin(['33', '34', '44'])].copy()

    n_pre_dedup = len(d)
    if LMM_DEDUPLICATE_SHARED_BOUNDARIES and dx_group in ['Overall', 'MCI + Dementia']:
        d = (
            d.sort_values(
                ['BIDS_SUBJECT', 'AGE_VISIT', 'WINDOW_INDEX', 'TIME'],
                kind='mergesort',
            )
            .drop_duplicates(subset=['SOURCE_OBS_ID'], keep='first')
            .copy()
        )
    n_removed = n_pre_dedup - len(d)
    if d.empty:
        prep = _empty_prep()
        prep['n_removed_shared_boundary_rows'] = int(n_removed)
        prep['n_units_before_requirements'] = n_before
        return d, prep

    valid_window_counts = d.groupby('WINDOW_UID').size()
    valid_windows = valid_window_counts[valid_window_counts >= 2].index
    d = d[d['WINDOW_UID'].isin(valid_windows)].copy()
    if d.empty:
        prep = _empty_prep()
        prep['n_removed_shared_boundary_rows'] = int(n_removed)
        prep['n_units_before_requirements'] = n_before
        return d, prep

    x_mean = float(d[['WINDOW_UID', 'X_RAW']].drop_duplicates()['X_RAW'].mean())
    age_mean = float(d[['WINDOW_UID', 'AGE_BL']].drop_duplicates()['AGE_BL'].mean())
    d['X_CENTERED'] = d['X_RAW'] - x_mean
    d['X_CENTERED_SQ'] = d['X_CENTERED'] ** 2
    d['AGE_BL_CENTERED'] = d['AGE_BL'] - age_mean
    return d, {
        'n_removed_shared_boundary_rows': int(n_removed),
        'x_centering_mean': x_mean,
        'baseline_age_centering_mean': age_mean,
        'sex_levels': ','.join(sorted(d['SEX_MODEL'].astype(str).unique())),
        'n_units_before_requirements': n_before,
        'n_units_after_requirements': int(d['WINDOW_UID'].nunique()),
    }


def model_formula(model_spec, dx_group, d):
    x_degree = model_spec['x_degree']
    x_e4_interaction = bool(model_spec['x_e4_interaction'])
    quadratic_x_e4_interaction = bool(model_spec.get('quadratic_x_e4_interaction', False))

    if x_degree == 1 and x_e4_interaction:
        rhs = 'TIME * X_CENTERED * E4_DOSE'
    elif x_degree == 2 and x_e4_interaction and quadratic_x_e4_interaction:
        rhs = 'TIME * (X_CENTERED + X_CENTERED_SQ) * E4_DOSE'
    elif x_degree == 2 and not x_e4_interaction and quadratic_x_e4_interaction:
        rhs = 'TIME * X_CENTERED + TIME * X_CENTERED_SQ * E4_DOSE'
    elif x_degree == 2 and x_e4_interaction and not quadratic_x_e4_interaction:
        rhs = 'TIME * (X_CENTERED + X_CENTERED_SQ) + TIME * X_CENTERED * E4_DOSE'
    else:
        raise ValueError(
            f"Unsupported model combination: x_degree={x_degree}, "
            f"x_e4_interaction={x_e4_interaction}, "
            f"quadratic_x_e4_interaction={quadratic_x_e4_interaction}"
        )

    formula = f'Y_OBS ~ {rhs} + AGE_BL_CENTERED + C(SEX_MODEL)'
    add_dx = (
        dx_group == 'Overall' and ADJUST_OVERALL_FOR_DX_MAIN_EFFECT
    ) or (
        dx_group == 'MCI + Dementia' and ADJUST_MCI_DEMENTIA_FOR_DX_MAIN_EFFECT
    )
    if add_dx and d['DX_LABEL'].nunique() > 1:
        formula += ' + C(DX_LABEL)'
    return formula


def _design_rank_info(exog):
    arr = np.asarray(exog, dtype=float)
    rank = int(np.linalg.matrix_rank(arr, tol=DESIGN_RANK_TOLERANCE))
    n_cols = int(arr.shape[1])
    condition_number = float(np.linalg.cond(arr)) if arr.size else np.nan
    return rank, n_cols, condition_number


def _base_fit_info(model_spec, prep, d):
    unit_count = int(d['WINDOW_UID'].nunique()) if not d.empty and 'WINDOW_UID' in d else 0
    subj_count = int(d['BIDS_SUBJECT'].nunique()) if not d.empty else 0
    genotype_subject_counts = {}
    if not d.empty:
        genotype_subject_counts = (
            d[['BIDS_SUBJECT', 'APOE']].drop_duplicates()
            .groupby('APOE')['BIDS_SUBJECT'].nunique().to_dict()
        )
    return {
        'model_key': model_spec['key'],
        'model_family': 'LMM',
        'x_degree': model_spec['x_degree'],
        'x_e4_interaction': bool(model_spec['x_e4_interaction']),
        'quadratic_x_e4_interaction': bool(model_spec.get('quadratic_x_e4_interaction', False)),
        'status': 'not_fit',
        'reason': '',
        'formula': '',
        'covariance_or_random_effects': 'subject random intercept',
        'optimizer': '',
        'converged': False,
        'reml': bool(LMM_REML),
        'n_observations': int(len(d)),
        'n_subjects': subj_count,
        'n_model_units': unit_count,
        'n_genotypes': int(d['APOE'].nunique()) if not d.empty else 0,
        'n_APOE33_subjects': int(genotype_subject_counts.get('33', 0)),
        'n_APOE34_subjects': int(genotype_subject_counts.get('34', 0)),
        'n_APOE44_subjects': int(genotype_subject_counts.get('44', 0)),
        'n_removed_shared_boundary_rows': prep['n_removed_shared_boundary_rows'],
        'n_units_before_requirements': prep['n_units_before_requirements'],
        'n_units_after_requirements': prep['n_units_after_requirements'],
        'x_centering_mean': prep['x_centering_mean'],
        'baseline_age_centering_mean': prep['baseline_age_centering_mean'],
        'sex_levels': prep['sex_levels'],
        'apoe_reference': 'E4_DOSE: APOE33=0, APOE34=1, APOE44=2',
        'apoe_parameterization': (
            'linear_E4_DOSE_with_linear_and_quadratic_X_interactions'
            if model_spec['x_e4_interaction'] and model_spec.get('quadratic_x_e4_interaction', False) else
            'linear_E4_DOSE_with_linear_X_interaction_only'
            if model_spec['x_e4_interaction'] else
            'linear_E4_DOSE_with_quadratic_X_interaction_only'
            if model_spec.get('quadratic_x_e4_interaction', False) else
            'linear_E4_DOSE_without_X_interaction'
        ),
        'x_function': 'linear' if model_spec['x_degree'] == 1 else 'quadratic_hierarchical',
        'design_rank': np.nan,
        'design_columns': np.nan,
        'condition_number': np.nan,
        'warnings': '',
        'random_intercept_variance': np.nan,
    }


def _validate_formula_contract(result, model_spec):
    names = {str(x) for x in result.model.exog_names}
    if not any('TIME' in x for x in names):
        raise RuntimeError('MixedLM design contains no TIME term.')
    if any('APOE44' in x for x in names):
        raise RuntimeError('APOE44-specific terms are prohibited in the E4-dose-only models.')

    has_quad = any('X_CENTERED_SQ' in x for x in names)
    if model_spec['x_degree'] == 1 and has_quad:
        raise RuntimeError('Linear-X model unexpectedly contains quadratic X terms.')
    if model_spec['x_degree'] == 2:
        required = {'X_CENTERED_SQ', 'TIME:X_CENTERED_SQ'}
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f'Quadratic-X model is missing required terms: {missing}')

    linear_x_e4_terms = {
        'X_CENTERED:E4_DOSE',
        'TIME:X_CENTERED:E4_DOSE',
    }
    quadratic_x_e4_terms = {
        'X_CENTERED_SQ:E4_DOSE',
        'TIME:X_CENTERED_SQ:E4_DOSE',
    }
    has_linear_x_e4 = bool(model_spec['x_e4_interaction'])
    has_quadratic_x_e4 = bool(model_spec.get('quadratic_x_e4_interaction', False))

    if has_linear_x_e4:
        missing = sorted(linear_x_e4_terms - names)
        if missing:
            raise RuntimeError(f'Linear X-by-E4 interaction is missing required terms: {missing}')
    else:
        present = sorted(linear_x_e4_terms & names)
        if present:
            raise RuntimeError(
                'Model unexpectedly contains linear biomarker-by-E4 terms: '
                f'{present}'
            )

    if has_quadratic_x_e4:
        if model_spec['x_degree'] != 2:
            raise RuntimeError('Quadratic X-by-E4 interaction requested for a non-quadratic model.')
        missing = sorted(quadratic_x_e4_terms - names)
        if missing:
            raise RuntimeError(f'Quadratic X-by-E4 interaction is missing required terms: {missing}')
    else:
        present = sorted(quadratic_x_e4_terms & names)
        if present:
            raise RuntimeError(
                'Model constrains common curvature across E4 dose but unexpectedly '
                f'contains X-squared-by-E4 terms: {present}'
            )
    if 'TIME:E4_DOSE' not in names:
        raise RuntimeError(
            'Every E4-dose model must retain TIME:E4_DOSE so epsilon4 dose can '
            'shift annual change at the centered predictor value.'
        )

    offending = [
        x for x in names
        if 'TIME' in x and ('SEX_MODEL' in x or 'DX_LABEL' in x or 'AGE_BL_CENTERED' in x)
    ]
    if offending:
        raise RuntimeError(
            'Annual-rate contrasts require age, sex, and diagnosis to remain '
            f'main-effect adjustments; prohibited TIME interactions={offending}'
        )


def fit_panel_model(long_data, spec, dx_group, model_spec):
    d, prep = prepare_lmm_panel(long_data, spec, dx_group, model_spec)
    info = _base_fit_info(model_spec, prep, d)

    if len(d) < LMM_MIN_OBSERVATIONS:
        info['reason'] = f'fewer than {LMM_MIN_OBSERVATIONS} repeated observations'
        return None, d, info
    if d['BIDS_SUBJECT'].nunique() < LMM_MIN_SUBJECTS:
        info['reason'] = f'fewer than {LMM_MIN_SUBJECTS} unique subjects'
        return None, d, info
    if d['APOE'].nunique() < 2:
        info['reason'] = 'fewer than two APOE genotype groups'
        return None, d, info

    formula = model_formula(model_spec, dx_group, d)
    info['formula'] = formula
    try:
        model = smf.mixedlm(
            formula,
            data=d,
            groups=d['BIDS_SUBJECT'],
            re_formula='1',
        )
        rank, n_cols, cond = _design_rank_info(model.exog)
        info.update({
            'design_rank': rank,
            'design_columns': n_cols,
            'condition_number': cond,
        })
        if rank < n_cols:
            info['reason'] = f'rank-deficient fixed-effect design ({rank}/{n_cols})'
            return None, d, info

        fit_attempts = []
        result = None
        optimizer_chains = [LMM_OPTIMIZERS, ['bfgs', 'powell']]
        for chain in optimizer_chains:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    candidate = model.fit(
                        reml=LMM_REML,
                        method=chain,
                        maxiter=LMM_MAXITER,
                        disp=False,
                    )
                warning_text = ' | '.join(sorted({str(w.message) for w in caught}))
                finite_fe = np.all(np.isfinite(np.asarray(candidate.fe_params, dtype=float)))
                fe_names = candidate.fe_params.index
                cov_fe = candidate.cov_params().loc[fe_names, fe_names]
                finite_cov = np.all(np.isfinite(cov_fe.to_numpy(dtype=float)))
                converged = bool(getattr(candidate, 'converged', False))
                fit_attempts.append(
                    f'{"->".join(chain)}:converged={converged},'
                    f'finite_fe={finite_fe},finite_cov={finite_cov},'
                    f'warnings={warning_text or "none"}'
                )
                if converged and finite_fe and finite_cov:
                    result = candidate
                    successful_chain = chain
                    break
            except Exception as fit_exc:
                fit_attempts.append(
                    f'{"->".join(chain)}:{type(fit_exc).__name__}:{fit_exc}'
                )

        if result is None:
            info['status'] = 'failed'
            info['reason'] = fit_attempts[-1] if fit_attempts else 'all MixedLM attempts failed'
            info['warnings'] = ' || '.join(fit_attempts)
            return None, d, info

        warning_text = ' || '.join(fit_attempts)

        _validate_formula_contract(result, model_spec)
        cov_re = np.asarray(result.cov_re, dtype=float)
        ri_var = float(cov_re.flat[0]) if cov_re.size else np.nan
        info.update({
            'status': 'ok',
            'reason': '',
            'optimizer': '->'.join(successful_chain),
            'converged': True,
            'warnings': warning_text,
            'random_intercept_variance': ri_var,
        })
        return result, d, info
    except Exception as exc:
        info['status'] = 'failed'
        info['reason'] = f'{type(exc).__name__}:{exc}'
        return None, d, info


def fixed_effect_bundle(result, model_spec):
    beta = result.fe_params.copy()
    cov = result.cov_params().loc[beta.index, beta.index].copy()
    bse = result.bse_fe.reindex(beta.index)
    pvals = result.pvalues.reindex(beta.index)
    return beta, cov, bse, pvals


def _critical_value(result, model_spec):
    return 1.959963984540054


def design_weight_matrix(result, model_data, model_spec, x_values, apoe):
    xs = np.asarray(x_values, dtype=float)
    beta, _, _, _ = fixed_effect_bundle(result, model_spec)
    names = list(beta.index)
    if xs.size == 0 or str(apoe) not in APOE_E4_DOSE:
        return np.empty((0, len(names)))

    x_mean = float(model_data.attrs['x_centering_mean'])
    xc = xs - x_mean
    x2 = xc ** 2
    e4 = float(APOE_E4_DOSE[str(apoe)])
    W = np.zeros((xs.size, len(names)), dtype=float)
    col = {name: j for j, name in enumerate(names)}

    def put(term, values):
        j = col.get(term)
        if j is not None:
            W[:, j] = values

    ones = np.ones(xs.size, dtype=float)
    put('TIME', ones)
    put('TIME:X_CENTERED', xc)
    put('TIME:X_CENTERED_SQ', x2)
    put('TIME:E4_DOSE', e4 * ones)
    put('TIME:X_CENTERED:E4_DOSE', e4 * xc)
    put('TIME:X_CENTERED_SQ:E4_DOSE', e4 * x2)
    return W


def predicted_curve(result, model_data, model_spec, x_values, apoe):
    xs = np.asarray(x_values, dtype=float)
    beta, cov, _, _ = fixed_effect_bundle(result, model_spec)
    W = design_weight_matrix(result, model_data, model_spec, xs, apoe)
    if W.shape[0] == 0:
        return pd.DataFrame(columns=['x_value', 'predicted_rate', 'ci_lo', 'ci_hi'])
    estimate = W @ beta.to_numpy(dtype=float)
    variance = np.einsum('ij,jk,ik->i', W, cov.to_numpy(dtype=float), W)
    se = np.sqrt(np.maximum(variance, 0.0))
    crit = _critical_value(result, model_spec)
    return pd.DataFrame({
        'x_value': xs,
        'predicted_rate': estimate,
        'ci_lo': estimate - crit * se,
        'ci_hi': estimate + crit * se,
    })


def fixed_contrast(result, model_spec, weights):
    beta, cov, _, _ = fixed_effect_bundle(result, model_spec)
    w = pd.Series(0.0, index=beta.index)
    if isinstance(weights, pd.Series):
        w.loc[w.index.intersection(weights.index)] = weights.reindex(w.index).fillna(0.0)
    else:
        arr = np.asarray(weights, dtype=float)
        if arr.size != len(w):
            raise ValueError('Contrast vector length does not match fixed effects.')
        w.iloc[:] = arr
    estimate = float(w @ beta)
    wv = w.to_numpy(dtype=float)
    variance = float(wv @ cov.to_numpy(dtype=float) @ wv)
    se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
    if not np.isfinite(se) or se <= 0:
        return {
            'estimate': estimate, 'standard_error': np.nan, 'statistic': np.nan,
            'p_value': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
            'reference_distribution': '',
        }
    stat = estimate / se
    p = float(2 * norm.sf(abs(stat)))
    crit = 1.959963984540054
    return {
        'estimate': estimate,
        'standard_error': se,
        'statistic': float(stat),
        'p_value': p,
        'ci_lo': float(estimate - crit * se),
        'ci_hi': float(estimate + crit * se),
        'reference_distribution': 'normal',
    }


def _weight_at_x(result, model_data, model_spec, x_raw, apoe):
    W = design_weight_matrix(
        result, model_data, model_spec, np.asarray([x_raw], dtype=float), apoe
    )
    beta, _, _, _ = fixed_effect_bundle(result, model_spec)
    if W.shape[0] == 0:
        return pd.Series(0.0, index=beta.index)
    return pd.Series(W[0], index=beta.index)


def _linear_combo(*items):
    series = [s for _, s in items]
    if not series:
        return pd.Series(dtype=float)
    idx = series[0].index
    out = pd.Series(0.0, index=idx)
    for coef, s in items:
        out = out + float(coef) * s.reindex(idx, fill_value=0.0)
    return out


def assert_curve_contract(result, model_data, model_spec, apoe, curve):
    if curve.empty:
        return
    x_mean = float(model_data.attrs['x_centering_mean'])
    direct = fixed_contrast(
        result, model_spec, _weight_at_x(result, model_data, model_spec, x_mean, apoe)
    )['estimate']
    plotted = float(
        predicted_curve(result, model_data, model_spec, [x_mean], apoe)
        ['predicted_rate'].iloc[0]
    )
    if np.isfinite(direct) and np.isfinite(plotted):
        scale = max(1.0, abs(direct))
        if abs(direct - plotted) > CURVE_CONTRAST_TOLERANCE * scale:
            raise RuntimeError(
                f'Curve/contrast mismatch for {model_spec["key"]} APOE{apoe}: '
                f'curve={plotted}, contrast={direct}'
            )


def assert_apoe_parameterization(result, model_data, model_spec):
    observed = set(model_data['APOE'].astype(str))
    if observed != {'33', '34', '44'}:
        return
    x_mean = float(model_data.attrs['x_centering_mean'])
    scale_x = float(model_data['X_RAW'].std(ddof=0))
    if not np.isfinite(scale_x) or scale_x <= 0:
        scale_x = 1.0
    xs = [x_mean - 0.5 * scale_x, x_mean, x_mean + 0.5 * scale_x]
    preds = {}
    for x in xs:
        pred = {
            g: fixed_contrast(
                result, model_spec, _weight_at_x(result, model_data, model_spec, x, g)
            )['estimate']
            for g in ['33', '34', '44']
        }
        preds[x] = pred
        deviation = pred['44'] - 2.0 * pred['34'] + pred['33']
        tol = CURVE_CONTRAST_TOLERANCE * max(1.0, *(abs(v) for v in pred.values()))
        if abs(deviation) > tol:
            raise RuntimeError(
                f'E4-dose model violates linear-dose identity at x={x}: '
                f'44-2*34+33={deviation}'
            )

    has_linear_x_e4 = bool(model_spec['x_e4_interaction'])
    has_quadratic_x_e4 = bool(model_spec.get('quadratic_x_e4_interaction', False))
    if has_linear_x_e4 and model_spec['x_degree'] == 2 and not has_quadratic_x_e4:
        for g in ['34', '44']:
            offsets = np.asarray([preds[x][g] - preds[x]['33'] for x in xs], dtype=float)
            second_difference = float(offsets[0] - 2.0 * offsets[1] + offsets[2])
            tol = CURVE_CONTRAST_TOLERANCE * max(1.0, float(np.max(np.abs(offsets))))
            if abs(second_difference) > tol:
                raise RuntimeError(
                    f'Linear-X-only E4 interaction model produced APOE-specific '
                    f'curvature for APOE{g}: second_difference={second_difference}'
                )
    elif not has_linear_x_e4 and has_quadratic_x_e4:
        for g in ['34', '44']:
            w = _linear_combo(
                (1.0, _rate_derivative_weight(result, model_data, model_spec, x_mean, g)),
                (-1.0, _rate_derivative_weight(result, model_data, model_spec, x_mean, '33')),
            )
            est = fixed_contrast(result, model_spec, w)['estimate']
            tol = CURVE_CONTRAST_TOLERANCE * max(1.0, abs(float(est)))
            if abs(float(est)) > tol:
                raise RuntimeError(
                    f'Curvature-only E4 interaction model produced a nonzero '
                    f'linear-slope difference at centered X for APOE{g}: {est}'
                )


def _rate_derivative_weight(result, model_data, model_spec, x_raw, apoe):
    x_sd = float(pd.to_numeric(model_data['X_RAW'], errors='coerce').std(ddof=0))
    if not np.isfinite(x_sd) or x_sd <= 0:
        x_sd = 1.0
    h = max(1e-6, 1e-5 * x_sd)
    return _linear_combo(
        (1.0 / (2.0 * h), _weight_at_x(result, model_data, model_spec, x_raw + h, apoe)),
        (-1.0 / (2.0 * h), _weight_at_x(result, model_data, model_spec, x_raw - h, apoe)),
    )


def _rate_curvature_weight(result, model_data, model_spec, x_raw, apoe):
    x_sd = float(pd.to_numeric(model_data['X_RAW'], errors='coerce').std(ddof=0))
    if not np.isfinite(x_sd) or x_sd <= 0:
        x_sd = 1.0
    h = max(1e-4, 1e-3 * x_sd)
    return _linear_combo(
        (1.0 / (h * h), _weight_at_x(result, model_data, model_spec, x_raw + h, apoe)),
        (-2.0 / (h * h), _weight_at_x(result, model_data, model_spec, x_raw, apoe)),
        (1.0 / (h * h), _weight_at_x(result, model_data, model_spec, x_raw - h, apoe)),
    )


def build_model_contrasts(result, model_data, fit_info, spec, dx_group, model_spec):
    rows = []
    x_mean = float(fit_info['x_centering_mean'])
    observed = [g for g in ['33', '34', '44'] if g in set(model_data['APOE'].astype(str))]
    rate_w = {g: _weight_at_x(result, model_data, model_spec, x_mean, g) for g in observed}
    deriv_w = {
        g: _rate_derivative_weight(result, model_data, model_spec, x_mean, g)
        for g in observed
    }
    curvature_w = {
        g: _rate_curvature_weight(result, model_data, model_spec, x_mean, g)
        for g in observed
    } if model_spec['x_degree'] == 2 else {}

    base = {
        'model_key': model_spec['key'],
        'model_family': 'LMM',
        'x_degree': model_spec['x_degree'],
        'x_e4_interaction': bool(model_spec['x_e4_interaction']),
        'quadratic_x_e4_interaction': bool(model_spec.get('quadratic_x_e4_interaction', False)),
        'section': spec['section'],
        'data_type': spec['data_type'],
        'subplot': dx_group,
        'x_variable': spec['xcol'],
        'x_label': spec['x_label'],
        'x_centering_mean': x_mean,
        'n_observations_model': int(len(model_data)),
        'n_subjects_model': int(model_data['BIDS_SUBJECT'].nunique()),
        'n_model_units': int(model_data['WINDOW_UID'].nunique()),
        'formula': fit_info['formula'],
    }

    for g in observed:
        rows.append({
            **base, 'APOE': g, 'quantity': 'annual_rate_at_mean_predictor',
            **{f'contrast_{k}': v for k, v in fixed_contrast(result, model_spec, rate_w[g]).items()},
        })
        rows.append({
            **base, 'APOE': g, 'quantity': 'annual_rate_derivative_per_X_unit_at_mean_predictor',
            **{f'contrast_{k}': v for k, v in fixed_contrast(result, model_spec, deriv_w[g]).items()},
        })
        if model_spec['x_degree'] == 2:
            rows.append({
                **base, 'APOE': g, 'quantity': 'annual_rate_curvature_per_X_squared',
                **{f'contrast_{k}': v for k, v in fixed_contrast(result, model_spec, curvature_w[g]).items()},
            })
        if '33' in observed and g != '33':
            rows.append({
                **base, 'APOE': g,
                'quantity': 'annual_rate_at_mean_predictor_difference_vs_APOE33',
                **{f'contrast_{k}': v for k, v in fixed_contrast(
                    result, model_spec,
                    _linear_combo((1.0, rate_w[g]), (-1.0, rate_w['33']))
                ).items()},
            })
            rows.append({
                **base, 'APOE': g,
                'quantity': 'annual_rate_derivative_difference_vs_APOE33',
                **{f'contrast_{k}': v for k, v in fixed_contrast(
                    result, model_spec,
                    _linear_combo((1.0, deriv_w[g]), (-1.0, deriv_w['33']))
                ).items()},
            })
            if model_spec['x_degree'] == 2:
                rows.append({
                    **base, 'APOE': g,
                    'quantity': 'annual_rate_curvature_difference_vs_APOE33',
                    **{f'contrast_{k}': v for k, v in fixed_contrast(
                        result, model_spec,
                        _linear_combo((1.0, curvature_w[g]), (-1.0, curvature_w['33']))
                    ).items()},
                })

    if set(observed) == {'33', '34', '44'}:
        derived = [
            (
                'linear_E4_dose_step_33_to_34_at_mean_predictor',
                _linear_combo((1, rate_w['34']), (-1, rate_w['33'])),
            ),
            (
                'E4_dose_linearity_identity_at_mean_predictor',
                _linear_combo((1, rate_w['44']), (-2, rate_w['34']), (1, rate_w['33'])),
            ),
            (
                'linear_E4_dose_step_in_X_derivative',
                _linear_combo((1, deriv_w['34']), (-1, deriv_w['33'])),
            ),
            (
                'E4_dose_linearity_identity_in_X_derivative',
                _linear_combo((1, deriv_w['44']), (-2, deriv_w['34']), (1, deriv_w['33'])),
            ),
        ]
        if model_spec['x_degree'] == 2:
            derived.extend([
                (
                    'linear_E4_dose_step_in_X_curvature',
                    _linear_combo((1, curvature_w['34']), (-1, curvature_w['33'])),
                ),
                (
                    'E4_dose_linearity_identity_in_X_curvature',
                    _linear_combo((1, curvature_w['44']), (-2, curvature_w['34']), (1, curvature_w['33'])),
                ),
            ])
        for quantity, weights in derived:
            result_row = fixed_contrast(result, model_spec, weights)
            rows.append({
                **base,
                'APOE': 'derived',
                'quantity': quantity,
                **{f'contrast_{k}': v for k, v in result_row.items()},
                'interpretation': (
                    'Linear epsilon4-dose step.' if 'step' in quantity
                    else 'Constrained to zero by the 0/1/2 linear epsilon4-dose parameterization.'
                ),
            })
    return rows


def record_axis_clipping(model_spec, spec, dx_group, apoe, xvals, yvals, curve):
    x_lim, _, y_lim, _ = panel_axis_limits(spec, dx_group)
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    x_out = (x < x_lim[0]) | (x > x_lim[1])
    y_out = (y < y_lim[0]) | (y > y_lim[1])
    curve_out = band_out = 0
    if curve is not None and not curve.empty:
        rate = curve['predicted_rate'].to_numpy(dtype=float)
        lo = curve['ci_lo'].to_numpy(dtype=float)
        hi = curve['ci_hi'].to_numpy(dtype=float)
        curve_out = int(((rate < y_lim[0]) | (rate > y_lim[1])).sum())
        band_out = int(((lo < y_lim[0]) | (hi > y_lim[1])).sum())
    AXIS_CLIPPING_ROWS.append({
        'model_key': model_spec['key'],
        'model_family': 'LMM',
        'x_degree': model_spec['x_degree'],
        'x_e4_interaction': bool(model_spec['x_e4_interaction']),
        'quadratic_x_e4_interaction': bool(model_spec.get('quadratic_x_e4_interaction', False)),
        'section': spec['section'],
        'tracer': spec.get('tracer') or 'PTAU',
        'subplot': dx_group,
        'x_variable': spec['xcol'],
        'y_variable': spec['ycol'],
        'APOE': apoe,
        'x_lim_lo': float(x_lim[0]),
        'x_lim_hi': float(x_lim[1]),
        'y_lim_lo': float(y_lim[0]),
        'y_lim_hi': float(y_lim[1]),
        'n_points': int(len(x)),
        'n_points_outside_x': int(x_out.sum()),
        'n_points_outside_y': int(y_out.sum()),
        'n_points_outside_either': int((x_out | y_out).sum()),
        'observed_x_min': float(x.min()) if len(x) else np.nan,
        'observed_x_max': float(x.max()) if len(x) else np.nan,
        'observed_y_min': float(y.min()) if len(y) else np.nan,
        'observed_y_max': float(y.max()) if len(y) else np.nan,
        'n_curve_grid_points_outside_y': curve_out,
        'n_curve_grid_points_with_band_outside_y': band_out,
    })


def _panel_source(spec):
    if spec['data_type'] == 'ptau':
        return ptau_slope_df, ptau_repeated_long
    return (
        taupet_by_tracer[spec['tracer']],
        taupet_repeated_by_tracer[spec['tracer']],
    )


def _model_unit_count(model_data, model_spec, apoe):
    d = model_data[model_data['APOE'] == apoe]
    if d.empty:
        return 0
    return int(d['WINDOW_UID'].nunique())


def run_one_figure(model_spec):
    subplot_rows = []
    curve_rows = []
    term_rows = []
    contrast_rows = []
    point_rows = []
    fit_rows = []

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300, facecolor='white')
    fig.text(
        0.5, 0.997, model_spec['title'], ha='center', va='top',
        fontsize=FONT_SIZE['title'] + 1, fontweight='bold',
    )

    for ri, spec in enumerate(ROW_SPECS, start=1):
        y_pos = ROW_BOTTOMS[ri]
        for di, dx_group in enumerate(dx_order):
            x_pos = START_X + di * (PANEL_W + COL_GAP)
            ax = fig.add_axes([x_pos, y_pos, PANEL_W, ROW_H])
            add_title_with_box(ax, dx_group, FONT_SIZE['title'])

            slope_source, repeated_source = _panel_source(spec)
            scatter = filter_dx_group(slope_source, dx_group)
            scatter = scatter.dropna(
                subset=[
                    'BIDS_SUBJECT', 'APOE', spec['xcol'], spec['ycol'],
                    'DX_LABEL', 'DX_GROUP',
                ]
            ).copy()

            result, model_data, fit_info = fit_panel_model(
                repeated_source, spec, dx_group, model_spec
            )
            if not model_data.empty:
                model_data.attrs['x_centering_mean'] = fit_info['x_centering_mean']

            fit_rows.append({
                'section': spec['section'],
                'section_label': SECTION_LABELS[spec['section']],
                'data_type': spec['data_type'],
                'tracer': spec.get('tracer') or 'PTAU',
                'subplot': dx_group,
                'x_variable': spec['xcol'],
                'x_label': spec['x_label'],
                'plotted_rate_variable': spec['ycol'],
                'plotted_rate_label': spec['y_label'],
                **fit_info,
            })

            for _, rr in scatter.iterrows():
                point_rows.append({
                    'model_key': model_spec['key'],
                    'section': spec['section'],
                    'data_type': spec['data_type'],
                    'subplot': dx_group,
                    'x_variable': spec['xcol'],
                    'y_variable': spec['ycol'],
                    'APOE': rr['APOE'],
                    'BIDS_SUBJECT': rr['BIDS_SUBJECT'],
                    'WINDOW_INDEX': rr.get('WINDOW_INDEX', np.nan),
                    'DX_LABEL': rr.get('DX_LABEL', ''),
                    'DX_FU_LABEL': rr.get('DX_FU_LABEL', ''),
                    'DX_TRANSITION': rr.get('DX_TRANSITION', ''),
                    'TRANSITION_DIRECTION': rr.get('TRANSITION_DIRECTION', ''),
                    'AGE_BL': rr.get('AGE_BL', np.nan),
                    'AGE_FU': rr.get('AGE_FU', np.nan),
                    'FOLLOWUP_YEARS': rr.get('FOLLOWUP_YEARS', np.nan),
                    'SEX_MODEL': rr.get('SEX_MODEL', ''),
                    'x_value': float(rr[spec['xcol']]),
                    'y_value': float(rr[spec['ycol']]),
                    'note': (
                        'Point is a diagnosis-window linear slope. The overlaid '
                        f'curve is from {model_spec["key"]}.'
                    ),
                })

            if result is not None:
                _validate_formula_contract(result, model_spec)
                assert_apoe_parameterization(result, model_data, model_spec)
                beta, _, bse, pvals = fixed_effect_bundle(result, model_spec)
                for term in beta.index:
                    term_rows.append({
                        'model_key': model_spec['key'],
                        'model_family': 'LMM',
                        'x_degree': model_spec['x_degree'],
                        'section': spec['section'],
                        'section_label': SECTION_LABELS[spec['section']],
                        'data_type': spec['data_type'],
                        'tracer': spec.get('tracer') or 'PTAU',
                        'subplot': dx_group,
                        'x_variable': spec['xcol'],
                        'x_centering_mean': fit_info['x_centering_mean'],
                        'baseline_age_centering_mean': fit_info['baseline_age_centering_mean'],
                        'formula': fit_info['formula'],
                        'term': term,
                        'coefficient': float(beta[term]),
                        'standard_error': float(bse[term]),
                        'p_value': float(pvals[term]),
                        'n_observations': fit_info['n_observations'],
                        'n_subjects': fit_info['n_subjects'],
                        'n_model_units': fit_info['n_model_units'],
                        'covariance_or_random_effects': fit_info['covariance_or_random_effects'],
                    })
                contrast_rows.extend(
                    build_model_contrasts(
                        result, model_data, fit_info, spec, dx_group, model_spec
                    )
                )
            else:
                print(
                    f"[{model_spec['key']} warning] section={spec['section']} "
                    f"subplot={dx_group} x={spec['xcol']}: {fit_info['reason']}"
                )

            for apoe in apoe_order:
                grp = scatter[scatter['APOE'] == apoe].copy()
                n_points = int(len(grp))
                n_subjects = int(grp['BIDS_SUBJECT'].nunique())
                x_stats = summarize_series(grp[spec['xcol']])
                y_stats = summarize_series(grp[spec['ycol']])
                n_model_units = _model_unit_count(model_data, model_spec, apoe)

                reason = ''
                if result is None:
                    reason = f'model_not_fitted:{fit_info["reason"]}'
                elif n_points < spec['min_n']:
                    reason = f'fewer_than_{spec["min_n"]}_scatter_points'
                elif n_model_units < MIN_GENOTYPE_MODEL_UNITS_FOR_CURVE:
                    reason = (
                        f'fewer_than_{MIN_GENOTYPE_MODEL_UNITS_FOR_CURVE}_'
                        'diagnosis_windows_in_model_sample'
                    )
                can_draw = reason == ''
                summary = {
                    'model_key': model_spec['key'],
                    'model_family': 'LMM',
                    'x_degree': model_spec['x_degree'],
                    'section': spec['section'],
                    'section_label': SECTION_LABELS[spec['section']],
                    'data_type': spec['data_type'],
                    'tracer': spec.get('tracer') or 'PTAU',
                    'subplot': dx_group,
                    'APOE': apoe,
                    'x_variable': spec['xcol'],
                    'x_label': spec['x_label'],
                    'y_variable': spec['ycol'],
                    'y_label': spec['y_label'],
                    'n_points': n_points,
                    'n_subjects_scatter': n_subjects,
                    'n_model_units': n_model_units,
                    'curve_drawn': can_draw,
                    'curve_suppressed_reason': reason,
                    'fit_status': fit_info['status'],
                    'formula': fit_info['formula'],
                }
                for k, v in x_stats.items():
                    summary[f'x_{k}'] = v
                for k, v in y_stats.items():
                    summary[f'y_{k}'] = v
                subplot_rows.append(summary)

                if n_points == 0:
                    record_axis_clipping(
                        model_spec, spec, dx_group, apoe, [], [], None
                    )
                    continue
                xvals = grp[spec['xcol']].to_numpy(dtype=float)
                yvals = grp[spec['ycol']].to_numpy(dtype=float)
                ax.scatter(
                    xvals, yvals, s=8, alpha=0.3,
                    color=palette_apoe[apoe], edgecolor='none', zorder=3,
                )
                if not can_draw:
                    record_axis_clipping(
                        model_spec, spec, dx_group, apoe, xvals, yvals, None
                    )
                    continue

                model_x = pd.to_numeric(
                    model_data.loc[model_data['APOE'] == apoe, 'X_RAW'],
                    errors='coerce',
                ).dropna()
                if model_x.empty:
                    continue
                grid = np.linspace(
                    float(model_x.min()), float(model_x.max()), MODEL_N_GRID
                )
                curve = predicted_curve(result, model_data, model_spec, grid, apoe)
                assert_curve_contract(result, model_data, model_spec, apoe, curve)
                ax.fill_between(
                    curve['x_value'].to_numpy(),
                    curve['ci_lo'].to_numpy(),
                    curve['ci_hi'].to_numpy(),
                    color=palette_apoe[apoe], alpha=0.18,
                    edgecolor='none', zorder=2,
                )
                ax.plot(
                    curve['x_value'].to_numpy(),
                    curve['predicted_rate'].to_numpy(),
                    color=palette_apoe[apoe], lw=2, zorder=4,
                )
                record_axis_clipping(
                    model_spec, spec, dx_group, apoe, xvals, yvals, curve
                )
                for _, cr in curve.iterrows():
                    curve_rows.append({
                        'model_key': model_spec['key'],
                        'model_family': 'LMM',
                        'x_degree': model_spec['x_degree'],
                        'section': spec['section'],
                        'section_label': SECTION_LABELS[spec['section']],
                        'data_type': spec['data_type'],
                        'tracer': spec.get('tracer') or 'PTAU',
                        'subplot': dx_group,
                        'APOE': apoe,
                        'x_variable': spec['xcol'],
                        'x_value': float(cr['x_value']),
                        'predicted_y': float(cr['predicted_rate']),
                        'ci_lo': float(cr['ci_lo']),
                        'ci_hi': float(cr['ci_hi']),
                        'formula': fit_info['formula'],
                        'x_centering_mean': fit_info['x_centering_mean'],
                    })

            set_panel_axes(ax, spec, dx_group)

    for letter in SECTION_ORDER:
        first_row = SECTION_ROWS[letter][0]
        add_section_label(
            fig,
            ROW_BOTTOMS[first_row] + ROW_H + ROW_LABEL_Y_OFFSET,
            SECTION_LABELS[letter],
            START_X,
        )

    legend_elements = [
        Line2D([0], [0], color=palette_apoe[g], lw=2, label=apoe_labels[g])
        for g in apoe_order
    ]
    for letter in SECTION_ORDER:
        last_row = SECTION_ROWS[letter][-1]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, ROW_BOTTOMS[last_row] - LEGEND_GAP),
            ncol=3,
            frameon=False,
            fontsize=FONT_SIZE['legend'],
        )

    figure_path = os.path.join(OUTPUT_DIR, model_spec['filename'])
    fig.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {figure_path}')

    prefix = os.path.join(OUTPUT_DIR, f"FIG_S3_{model_spec['order']}_{model_spec['key']}")
    outputs = {
        'subplot_summary': pd.DataFrame(subplot_rows),
        'curve_grid': pd.DataFrame(curve_rows),
        'model_terms': pd.DataFrame(term_rows),
        'model_contrasts': pd.DataFrame(contrast_rows),
        'data_points': pd.DataFrame(point_rows),
        'fit_summary': pd.DataFrame(fit_rows),
    }
    for suffix, frame in outputs.items():
        path = f'{prefix}_{suffix}.csv'
        frame.to_csv(path, index=False)
        print(f'Saved: {path}')

    return figure_path


def main():
    global df, BIDS_SUBJECT_col_main, main_subject_lookup, ptau_long, BIDS_SUBJECT_col_ptau, ptau_subject_lookup, cross_sheet_subject_audit
    global TAU_SECTION_INFO, taupet_by_tracer, ptau_slope_df, ptau_repeated_long, taupet_repeated_by_tracer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MAIN_FILE = 'DIMI_DATASET.csv'
    PTAU_FILE = 'p-tau217.csv'

    if not os.path.exists(MAIN_FILE):
        raise FileNotFoundError(
            f'{MAIN_FILE} was not found in the working directory. It is required.'
        )
    if not os.path.exists(PTAU_FILE):
        print(
            f'[input] {PTAU_FILE} not found. Section a and baseline-p-tau217 panels '
            'will be empty; tau PET analyses that do not require p-tau can still run.'
        )

    if [m['order'] for m in MODEL_SPECS] != [1, 2, 3, 4]:
        raise RuntimeError('MODEL_SPECS must contain exactly the ordered figures 1-4.')
    if [m['x_degree'] for m in MODEL_SPECS] != [1, 2, 2, 2]:
        raise RuntimeError('MODEL_SPECS must contain linear-X followed by three quadratic-X models.')
    scopes = [
        (bool(m['x_e4_interaction']), bool(m.get('quadratic_x_e4_interaction', False)))
        for m in MODEL_SPECS
    ]
    if scopes != [(True, False), (True, True), (True, False), (False, True)]:
        raise RuntimeError(
            'MODEL_SPECS interaction order must be: linear X-by-E4; full quadratic '
            'X-by-E4; linear-X-only E4 interaction; curvature-only E4 interaction.'
        )

    print('Loading and harmonizing shared data...')
    df, BIDS_SUBJECT_col_main, main_subject_lookup = prepare_main_data(MAIN_FILE)
    ptau_long, BIDS_SUBJECT_col_ptau, ptau_subject_lookup = prepare_ptau_data(
        PTAU_FILE, main_lookup=main_subject_lookup
    )
    cross_sheet_subject_audit = build_cross_sheet_subject_audit(
        main_subject_lookup, ptau_subject_lookup
    )

    TAU_SECTION_INFO = {}
    taupet_by_tracer = {}
    for letter, tracer_key in TAU_TRACER_SECTIONS:
        cfg = dict(TAU_TRACER_CONFIG[tracer_key])
        cfg['key'] = tracer_key
        if cfg['suvr_col'] not in df.columns:
            raise KeyError(
                f"Section {letter} requires {cfg['suvr_col']!r} in {MAIN_FILE}."
            )
        TAU_SECTION_INFO[letter] = cfg
        tdf = compute_taupet_subject_slopes(
            df, BIDS_SUBJECT_col_main, region_col=cfg['suvr_col']
        )
        tdf = attach_ptau_baseline(
            tdf, ptau_long, BIDS_SUBJECT_col_ptau,
            window_years=PTAU_BL_WINDOW_YEARS,
        )
        tdf = attach_cl_baseline(
            tdf, df, BIDS_SUBJECT_col_main,
            window_years=CL_BL_WINDOW_YEARS,
        )
        taupet_by_tracer[tracer_key] = tdf

    ptau_slope_df = compute_ptau_subject_slopes(ptau_long, BIDS_SUBJECT_col_ptau)
    ptau_slope_df = attach_cl_baseline(
        ptau_slope_df, df, BIDS_SUBJECT_col_main,
        window_years=CL_BL_WINDOW_YEARS,
    )

    ptau_repeated_long = build_repeated_long_data(
        ptau_long, BIDS_SUBJECT_col_ptau, 'AGE_at_visit', 'zscore_2sd',
        ptau_slope_df, source_tag='PTAU',
    )
    taupet_repeated_by_tracer = {}
    for letter, tracer_key in TAU_TRACER_SECTIONS:
        cfg = TAU_SECTION_INFO[letter]
        taupet_repeated_by_tracer[tracer_key] = build_repeated_long_data(
            df, BIDS_SUBJECT_col_main, 'AGE', cfg['suvr_col'],
            taupet_by_tracer[tracer_key], source_tag=f'TAUPET_{tracer_key}',
        )

    print(
        f'  pTau window slopes: {len(ptau_slope_df):,}; '
        f'repeated observations: {len(ptau_repeated_long):,}'
    )
    for letter, tracer_key in TAU_TRACER_SECTIONS:
        print(
            f'  {tracer_key} window slopes: {len(taupet_by_tracer[tracer_key]):,}; '
            f'repeated observations: {len(taupet_repeated_by_tracer[tracer_key]):,}'
        )

    pd.DataFrame(DATA_FLOW_AUDIT_ROWS).to_csv(
        os.path.join(OUTPUT_DIR, 'FIG_S3_data_flow_audit.csv'), index=False
    )
    cross_sheet_subject_audit.to_csv(
        os.path.join(OUTPUT_DIR, 'FIG_S3_cross_sheet_subject_audit.csv'), index=False
    )

    figure_paths = []
    for model_spec in MODEL_SPECS:
        print(f"\nRunning figure {model_spec['order']}: {model_spec['key']}")
        figure_paths.append(run_one_figure(model_spec))

    pd.DataFrame(LMM_WINDOW_AUDIT_ROWS).to_csv(
        os.path.join(OUTPUT_DIR, 'FIG_S3_model_window_inclusion_audit.csv'), index=False
    )
    clipping_audit = pd.DataFrame(AXIS_CLIPPING_ROWS)
    clipping_audit.to_csv(
        os.path.join(OUTPUT_DIR, 'FIG_S3_axis_clipping_audit.csv'), index=False
    )

    if not clipping_audit.empty:
        hidden = clipping_audit[clipping_audit['n_points_outside_either'] > 0]
        if not hidden.empty:
            print(
                f"\n[axis clipping] {int(hidden['n_points_outside_either'].sum()):,} "
                'point-panel occurrences fall outside fixed axes. Inspect '
                'FIG_S3_axis_clipping_audit.csv.'
            )
        else:
            print('\n[axis clipping] No scatter points fall outside the fixed axes.')

    manifest = pd.DataFrame([
        {
            'figure_order': m['order'],
            'model_key': m['key'],
            'model_family': 'LMM',
            'x_degree': m['x_degree'],
            'x_e4_interaction': bool(m['x_e4_interaction']),
            'quadratic_x_e4_interaction': bool(m.get('quadratic_x_e4_interaction', False)),
            'figure_file': m['filename'],
            'title': m['title'],
            'description': m['description'],
        }
        for m in MODEL_SPECS
    ])
    manifest.to_csv(os.path.join(OUTPUT_DIR, 'FIG_S3_four_LMM_model_manifest.csv'), index=False)

    print('\nFour LMM figures completed:')
    for path in figure_paths:
        print(f'  {path}')


if __name__ == '__main__':
    main()
