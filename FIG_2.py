"""
Figure 2. Plasma p-tau217 by Aβ burden, and tau PET visual read by Aβ burden
and plasma p-tau217, by clinical status and APOE haplotype.

Section a plots plasma p-tau217 z-scores against Aβ burden (Centiloid) for APOE
2/3, 3/3, 3/4 and 4/4, with LOESS curves and 95% bootstrap confidence bands
(1,000 resamples). Section b shows the percentage of p-tau217-positive
participants (z ≥ 1.15) in 10-Centiloid bins; markers indicate bins in which
haplotypes differ (chi-square test across haplotypes and pairwise Fisher exact
tests, Holm-corrected). Section c shows the proportion of positive
[18F]flortaucipir visual reads across Centiloid bins and across p-tau217
z-score bins for APOE 3/3, 3/4 and 4/4, with Wilson 95% confidence intervals.
Participants with co-pathology (COPAT = 1) are excluded and p-tau217 must lie
within 365 days of the PET scan. Centiloid and tau PET values come from input
datasets in which the amyloid and tau PET scans were already matched in time,
so no additional amyloid-to-tau PET interval filter is applied here. All
sections use each participant's earliest eligible visit; the Overall column has
one visit per participant and the CU, MCI and Dementia columns one visit per
participant and diagnosis. Points and bars are shown for cells with at least 5
participants.

Inputs:  p-tau217.csv, DIMI_DATASET_with_visual_read.csv,
         DIMI_with_ptau217_with_visual_read.csv
Outputs: FIG_2_ptau217_visual_read.png and CSV tables with sample selection,
         test results, LOESS estimates, bin proportions and descriptive
         statistics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.nonparametric.smoothers_lowess import lowess


PTAU_FILE = Path("p-tau217.csv")
VISUAL_READ_CL_FILE = Path("DIMI_DATASET_with_visual_read.csv")
VISUAL_READ_PTAU_FILE = Path("DIMI_with_ptau217_with_visual_read.csv")
OUTPUT_DIR = Path(".")

OUTPUT_PREFIX = "FIG_2"

MAX_DAYS_PTAU_PET = 365
PTAU_POS_THRESHOLD = 1.15
RANDOM_SEED = 42
ALPHA = 0.05

PTAU_VISIT_KEEP = "first"
MIN_N = 5
MIN_N_TEST = 5
FISHER_ALTERNATIVE = "two-sided"
LOESS_FRAC = 0.75
LOESS_N_BOOT = 1000
LOESS_GRID_N = 150

VISUAL_READ_OUTCOME_COL = "visual_read_binary_all"
VISUAL_READ_APOE = ["33", "34", "44"]
MATCH_FIG_S2_ELIGIBILITY = True
VISUAL_READ_SHOW_SIGNIFICANCE = False
MIN_BIN_N = 5
E2_GENOTYPES = ["22", "23", "24"]

CANONICALIZE_APOE_ORDER = True


FONT_SIZE = {
    "panel_label": 20,
    "title": 18,
    "axis_label": 16,
    "tick": 14,
    "legend": 18,
    "n_annot": 9,
}
TITLE_BAND_PTS = 25

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
    "axes.titleweight": "regular",
    "font.size": FONT_SIZE["tick"],
    "axes.labelsize": FONT_SIZE["axis_label"],
    "xtick.labelsize": FONT_SIZE["tick"],
    "ytick.labelsize": FONT_SIZE["tick"],
})

RNG = np.random.default_rng(RANDOM_SEED)

PALETTE_APOE = {
    "44": "#C7353E",
    "34": "#E0B04A",
    "33": "#389F42",
    "23": "#2E86C1",
}
APOE_ORDER = ["23", "33", "34", "44"]
APOE_LABELS = {
    "44": r"$\it{APOE}$4/4",
    "34": r"$\it{APOE}$3/4",
    "33": r"$\it{APOE}$3/3",
    "23": r"$\it{APOE}$2/3",
}
E4_DOSE = {"23": 0.0, "33": 0.0, "34": 1.0, "44": 2.0}

DX_MAP = {1: "CU", 2: "MCI", 3: "Dementia"}
DX_ORDER = ["CU", "MCI", "Dementia"]
OVERALL_LABEL = "Overall"
PANEL_ORDER = [OVERALL_LABEL, *DX_ORDER]
PANEL_SAMPLE_RULE = {
    OVERALL_LABEL: "one eligible row per BIDS_SUBJECT",
    "CU": "one eligible row per BIDS_SUBJECT x DX",
    "MCI": "one eligible row per BIDS_SUBJECT x DX",
    "Dementia": "one eligible row per BIDS_SUBJECT x DX",
}

CL10_EDGES = [-np.inf, 0] + list(range(10, 101, 10)) + [np.inf]
CL10_LABELS = ["<0", "0–10"] + [f"{x}–{x + 10}" for x in range(10, 100, 10)] + [">100"]
CL10_POS = np.arange(len(CL10_LABELS), dtype=float)

CL_BIN_EDGES = [-np.inf, 25, 50, 75, 100, np.inf]
CL_BIN_LABELS = ["≤25", "25–50", "50–75", "75–100", ">100"]

ZSCORE_BIN_EDGES = [-np.inf, 0, 1.5, 3.0, 4.5, np.inf]
ZSCORE_BIN_LABELS = ["<0", "0–<1.5", "1.5–<3.0", "3.0–<4.5", "≥4.5"]

PTAU_POS_COL = "PTAU217_POSITIVE"

PTAU_SERIES_STYLE = {
    "linestyle": "-",
    "marker": "o",
    "fill": True,
    "color_marker": "#222222",
}

DATE_SORT_CANDIDATES = [
    "PET_DATE", "MK_DATE", "TAU_PET_DATE", "SCAN_DATE", "EXAM_DATE",
    "DATE", "SESSION_DATE", "VISIT_DATE", "AcquisitionDate", "acquisition_date",
]

SECTION_LABELS = {
    0: (
        r"a. Plasma p-tau217 level by Aβ burden, clinical status, and "
        r"$\it{APOE}$ haplotype"
    ),
    1: (
        r"b. Plasma p-tau217 positivity by Aβ burden, clinical status, and "
        r"$\it{APOE}$ haplotype"
    ),
    2: (
        r"c. Observed proportion of visual [$^{18}$F]flortaucipir positivity by "
        r"Aβ burden and plasma p-tau217 level, clinical status, and "
        r"$\it{APOE}$ haplotype"
    ),
}

FIG_W = 24
PANEL_W = 0.18
COL_GAP = 0.05

ROW_H_IN = {"loess": 2.30, "rate": 1.90, "bar": 1.90}
BOTTOM_MARGIN_IN = 1.75
TOP_MARGIN_IN = 1.05
GAP_WITHIN_SECTION_IN = 1.35
GAP_BETWEEN_A_AND_B_IN = 2.45
GAP_BETWEEN_B_AND_C_IN = 2.80
ROW_LABEL_Y_OFFSET_IN = 0.73
LEGEND_Y_OFFSET_IN = {0: -1.00, 1: -1.35, 2: -1.25}

ROW_LABEL_X_SHIFT = -0.05
ROW_LABEL_TEXT_GAP = 0.015

BAR_WIDTH = 0.10
BAR_SMALL_GAP = 0.02
BAR_GROUP_GAP = 0.15


def output_path(filename: str) -> Path:
    return OUTPUT_DIR / filename


FIGURE_FILE = output_path(f"{OUTPUT_PREFIX}.png")
SELECTION_AUDIT_FILE = output_path(f"{OUTPUT_PREFIX}_sample_selection_audit.csv")
SIGNIFICANCE_FILE = output_path(f"{OUTPUT_PREFIX}_significance_debug.csv")
DESCRIPTIVES_FILE = output_path(f"{OUTPUT_PREFIX}_sample_descriptives.csv")
LOESS_FILE = output_path(f"{OUTPUT_PREFIX}_loess_section_a.csv")
BIN_PROPORTIONS_FILE = output_path(f"{OUTPUT_PREFIX}_bin_proportions_section_c.csv")
SUBPLOT_SUMMARY_FILE = output_path(f"{OUTPUT_PREFIX}_LLM_subplot_summary.csv")
DETAILED_FILE = output_path(f"{OUTPUT_PREFIX}_detailed_results_for_LLM.csv")

ALL_OUTPUT_FILES = [
    FIGURE_FILE, SELECTION_AUDIT_FILE, SIGNIFICANCE_FILE, DESCRIPTIVES_FILE,
    LOESS_FILE, BIN_PROPORTIONS_FILE, SUBPLOT_SUMMARY_FILE, DETAILED_FILE,
]

CL_ANCHORS = np.array([-25, 0, 25, 50, 75, 100, 125, 150, 175, 200], dtype=float)


def validate_configuration() -> None:
    if PTAU_VISIT_KEEP not in {"first", "last"}:
        raise ValueError("PTAU_VISIT_KEEP must be 'first' or 'last'.")
    if not set(VISUAL_READ_APOE).issubset(set(APOE_ORDER)):
        raise ValueError(f"VISUAL_READ_APOE must be a subset of {APOE_ORDER}.")
    if len(set(VISUAL_READ_APOE)) != len(VISUAL_READ_APOE):
        raise ValueError("VISUAL_READ_APOE contains duplicates.")
    if not VISUAL_READ_APOE:
        raise ValueError("VISUAL_READ_APOE must contain at least one genotype.")
    if MIN_N < 1 or MIN_N_TEST < 2 or MIN_BIN_N < 1:
        raise ValueError("Minimum cell sizes must be positive and MIN_N_TEST >= 2.")
    if not 0 < LOESS_FRAC <= 1:
        raise ValueError("LOESS_FRAC must lie in (0, 1].")
    if LOESS_N_BOOT < 40:
        raise ValueError("LOESS_N_BOOT below 40 cannot support a 95% percentile CI.")


def normalize_apoe(values: pd.Series) -> pd.Series:
    raw = values.copy()
    result = pd.Series(pd.NA, index=raw.index, dtype="string")

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_mask = numeric.notna() & np.isclose(numeric, np.round(numeric))
    if numeric_mask.any():
        result.loc[numeric_mask] = (
            np.round(numeric.loc[numeric_mask]).astype("Int64").astype("string")
        )

    text_mask = result.isna() & raw.notna()
    if text_mask.any():
        cleaned = (
            raw.loc[text_mask].astype("string").str.lower()
            .str.replace("epsilon", "", regex=False)
            .str.replace("apoe", "", regex=False)
            .str.replace("ε", "", regex=False)
            .str.replace("e", "", regex=False)
            .str.replace(r"[^0-9]", "", regex=True)
        )
        result.loc[text_mask] = cleaned

    if CANONICALIZE_APOE_ORDER:
        two_digit = result.notna() & result.str.fullmatch(r"[0-9]{2}").fillna(False)
        if two_digit.any():
            result.loc[two_digit] = result.loc[two_digit].map(
                lambda value: "".join(sorted(value)), na_action="ignore"
            )
    return result


def normalize_sex(values: pd.Series) -> pd.Series:
    raw = values.copy()
    result = pd.Series(pd.NA, index=raw.index, dtype="Float64")

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_present = numeric.dropna()
    if not numeric_present.empty:
        observed = {float(v) for v in np.unique(numeric_present)}
        if observed.issubset({0.0, 1.0}):
            result.loc[numeric.notna()] = numeric.loc[numeric.notna()].astype(float)
        elif observed.issubset({1.0, 2.0}):
            mapped = numeric.map({1.0: 1.0, 2.0: 0.0})
            result.loc[mapped.notna()] = mapped.loc[mapped.notna()]

    text_mask = result.isna() & raw.notna()
    if text_mask.any():
        cleaned = raw.loc[text_mask].astype("string").str.strip().str.lower()
        male_tokens = {"m", "male", "man", "masc", "masculino"}
        female_tokens = {"f", "female", "woman", "fem", "feminino"}
        mapped = pd.Series(pd.NA, index=cleaned.index, dtype="Float64")
        mapped.loc[cleaned.isin(male_tokens)] = 1.0
        mapped.loc[cleaned.isin(female_tokens)] = 0.0
        result.loc[mapped.notna()] = mapped.loc[mapped.notna()]
    return result.astype(float)


def require_columns(df: pd.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{source_name}: missing required columns: {missing}")


def validate_binary(series: pd.Series, column_name: str, source_name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.notna() & ~values.isin([0, 1])
    if invalid.any():
        examples = sorted(values.loc[invalid].unique().tolist())[:10]
        raise ValueError(
            f"{source_name}: {column_name} must contain only 0, 1 or missing. "
            f"Invalid examples: {examples}"
        )
    return values


def apply_copat_filter(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "COPAT" not in df.columns:
        print(f"  {dataset_name}: no COPAT column; no co-pathology exclusion applied.")
        return df.copy()

    values = pd.to_numeric(df["COPAT"], errors="coerce")
    excluded = values.eq(1)

    work = df.loc[~excluded].copy()
    work["COPAT"] = values.loc[~excluded].fillna(0.0)

    print(
        f"  {dataset_name}: COPAT excluded {int(excluded.sum()):,} rows "
        f"(COPAT == 1); retained {int(values.isna().sum()):,} rows with missing "
        f"COPAT, recoded to 0."
    )
    return work


def bin_mask(data: pd.DataFrame, bin_column: str, label: str) -> pd.Series:
    return data[bin_column].astype("string").eq(label)


def _safe_float(value) -> float:
    try:
        return np.nan if pd.isna(value) else float(value)
    except Exception:
        return np.nan


def numeric_summary(series: pd.Series, prefix: str) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    out = {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean": np.nan,
        f"{prefix}_sd": np.nan,
        f"{prefix}_median": np.nan,
        f"{prefix}_q1": np.nan,
        f"{prefix}_q3": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
    }
    if values.empty:
        return out
    out.update({
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q1": float(values.quantile(0.25)),
        f"{prefix}_q3": float(values.quantile(0.75)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    })
    return out


def association_summary(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    data = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    out = {
        "n_xy": int(len(data)),
        "pearson_r": np.nan,
        "spearman_rho": np.nan,
        "linear_slope": np.nan,
        "linear_intercept": np.nan,
        "linear_r2": np.nan,
    }
    if len(data) < 3 or data[x_col].nunique() < 2 or data[y_col].nunique() < 2:
        return out
    out["pearson_r"] = float(data[x_col].corr(data[y_col], method="pearson"))
    out["spearman_rho"] = float(data[x_col].corr(data[y_col], method="spearman"))
    slope, intercept = np.polyfit(
        data[x_col].to_numpy(float), data[y_col].to_numpy(float), 1
    )
    predicted = slope * data[x_col].to_numpy(float) + intercept
    ss_res = float(np.sum((data[y_col].to_numpy(float) - predicted) ** 2))
    ss_tot = float(np.sum((data[y_col].to_numpy(float) - data[y_col].mean()) ** 2))
    out.update({
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "linear_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    })
    return out


def holm_adjust(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return adjusted

    order = valid_idx[np.argsort(p[valid_idx])]
    m = len(order)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (max(0.0, 100.0 * (centre - half)), min(100.0, 100.0 * (centre + half)))


@dataclass
class PanelSample:
    overall: pd.DataFrame
    by_dx: pd.DataFrame
    source: str

    def panel(self, label: str) -> pd.DataFrame:
        if label == OVERALL_LABEL:
            return self.overall.copy()
        return self.by_dx.loc[self.by_dx["DX_LABEL"].eq(label)].copy()


def select_actual_rows(df: pd.DataFrame, subset: list[str], sort_col: str,
                       keep: str) -> pd.DataFrame:
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")
    return (
        df.sort_values(sort_col, kind="mergesort")
          .drop_duplicates(subset=subset, keep=keep)
          .reset_index(drop=True)
    )


def prep_ptau(df: pd.DataFrame) -> pd.DataFrame:
    source = PTAU_FILE.name
    required = [
        "BIDS_SUBJECT", "APOE", "DX", "AGE_at_visit", "CL",
        "zscore_2sd", "days_pTau_to_PET",
    ]
    require_columns(df, required, source)

    work = df.copy()
    work["APOE"] = normalize_apoe(work["APOE"])
    work = work.loc[work["APOE"].isin(APOE_ORDER)].copy()

    work["DX"] = pd.to_numeric(work["DX"], errors="coerce")
    work = work.loc[work["DX"].isin(DX_MAP)].copy()
    work["DX_LABEL"] = work["DX"].map(DX_MAP)

    work = apply_copat_filter(work, source)

    for column in ["AGE_at_visit", "CL", "zscore_2sd", "days_pTau_to_PET"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work["apoe_key"] = work["APOE"].astype("string")
    work["E4_DOSE"] = work["apoe_key"].map(E4_DOSE).astype(float)
    work["abs_days_pTau_to_PET"] = work["days_pTau_to_PET"].abs()
    work[PTAU_POS_COL] = np.where(
        work["zscore_2sd"].notna(),
        (work["zscore_2sd"] >= PTAU_POS_THRESHOLD).astype(float),
        np.nan,
    )
    return work


def build_ptau_sample() -> tuple[PanelSample, dict]:
    raw = pd.read_csv(PTAU_FILE)
    prepared = prep_ptau(raw)
    eligible = (
        prepared
        .loc[prepared["abs_days_pTau_to_PET"].le(MAX_DAYS_PTAU_PET)]
        .dropna(subset=[
            "BIDS_SUBJECT", "DX_LABEL", "AGE_at_visit", "CL",
            "zscore_2sd", "E4_DOSE", PTAU_POS_COL,
        ])
        .copy()
    )
    overall = select_actual_rows(eligible, ["BIDS_SUBJECT"], "AGE_at_visit", PTAU_VISIT_KEEP)
    by_dx = select_actual_rows(
        eligible, ["BIDS_SUBJECT", "DX_LABEL"], "AGE_at_visit", PTAU_VISIT_KEEP
    )
    for frame in (overall, by_dx):
        frame["CL10_bin"] = pd.cut(
            frame["CL"], bins=CL10_EDGES, labels=CL10_LABELS,
            include_lowest=True, ordered=True,
        )
    audit = {
        "n_source_rows": int(len(raw)),
        "n_after_basic_preparation": int(len(prepared)),
        "n_eligible_before_dedup": int(len(eligible)),
        "n_overall": int(len(overall)),
        "n_by_dx": int(len(by_dx)),
    }
    return PanelSample(overall=overall, by_dx=by_dx, source=PTAU_FILE.name), audit


def choose_dedup_sort(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in DATE_SORT_CANDIDATES:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                return parsed, column
    if "AGE" in df.columns:
        return pd.to_numeric(df["AGE"], errors="coerce"), "AGE"
    return pd.Series(np.arange(len(df)), index=df.index), "row_order"


def deduplicate_first_visit(df: pd.DataFrame, dataset_name: str,
                       keys: list[str]) -> tuple[pd.DataFrame, str, int]:
    before = len(df)
    work = df.copy()
    sort_values, sort_source = choose_dedup_sort(work)
    work["_DEDUP_SORT"] = sort_values
    work["_ORIGINAL_ORDER"] = np.arange(len(work))
    work = (
        work.sort_values([*keys, "_DEDUP_SORT", "_ORIGINAL_ORDER"], na_position="last")
        .drop_duplicates(keys, keep="first")
        .drop(columns=["_DEDUP_SORT", "_ORIGINAL_ORDER"])
        .reset_index(drop=True)
    )
    removed = before - len(work)
    print(
        f"  {dataset_name}: removed {removed:,} longitudinal rows within "
        f"{' × '.join(keys)} using {sort_source}."
    )
    return work, sort_source, removed


def prepare_visual_read_dataset(df: pd.DataFrame, dataset_name: str,
                                include_ptau: bool) -> pd.DataFrame:
    required = ["BIDS_SUBJECT", "APOE", "DX", "AGE", "SEX",
                VISUAL_READ_OUTCOME_COL]
    required.extend(["zscore_2sd", "days_diff"] if include_ptau else ["CL"])
    require_columns(df, required, dataset_name)

    work = df.copy()
    work["APOE"] = normalize_apoe(work["APOE"])
    work["DX"] = pd.to_numeric(work["DX"], errors="coerce")
    work["SEX_BINARY"] = normalize_sex(work["SEX"])
    for column in ["AGE", "CL", "zscore_2sd", "days_diff"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    work = apply_copat_filter(work, dataset_name)

    genotyped = work.loc[work["APOE"].notna() & work["DX"].isin(DX_MAP)]
    excluded = genotyped["APOE"].isin(
        [g for g in E2_GENOTYPES if g not in VISUAL_READ_APOE]
    )
    if excluded.any():
        counts = genotyped.loc[excluded, "APOE"].value_counts().to_dict()
        print(
            f"  {dataset_name}: excluded {int(excluded.sum()):,} epsilon-2 carrier "
            f"rows ({counts}); VISUAL_READ_APOE = {VISUAL_READ_APOE}."
        )

    work = work.loc[work["APOE"].isin(VISUAL_READ_APOE) & work["DX"].isin(DX_MAP)].copy()
    work["DX_LABEL"] = work["DX"].map(DX_MAP)
    work["apoe_key"] = work["APOE"].astype("string")
    work["E4_DOSE"] = work["apoe_key"].map(E4_DOSE).astype(float)

    work[VISUAL_READ_OUTCOME_COL] = validate_binary(
        work[VISUAL_READ_OUTCOME_COL], VISUAL_READ_OUTCOME_COL, dataset_name
    )
    work["positive"] = work[VISUAL_READ_OUTCOME_COL]

    if include_ptau:
        work["abs_days_diff"] = work["days_diff"].abs()
    return work


def build_visual_read_sample(path: Path, include_ptau: bool, bin_column: str,
                             bin_source: str, bin_edges: list,
                             bin_labels: list[str],
                             right: bool) -> tuple[PanelSample, dict]:
    raw = pd.read_csv(path)
    prepared = prepare_visual_read_dataset(raw, path.name, include_ptau)

    required_present = [bin_source, "positive", "AGE"]
    if MATCH_FIG_S2_ELIGIBILITY:
        required_present.append("SEX_BINARY")

    eligible = prepared
    if include_ptau:
        eligible = eligible.loc[eligible["abs_days_diff"].le(MAX_DAYS_PTAU_PET)]
    eligible = eligible.dropna(subset=required_present).copy()

    overall, sort_source, removed_overall = deduplicate_first_visit(
        eligible, f"{path.name} [Overall]", ["BIDS_SUBJECT"]
    )
    by_dx, _, removed_by_dx = deduplicate_first_visit(
        eligible, f"{path.name} [by DX]", ["BIDS_SUBJECT", "DX_LABEL"]
    )
    for frame in (overall, by_dx):
        frame[bin_column] = pd.cut(
            frame[bin_source], bins=bin_edges, labels=bin_labels,
            include_lowest=True, right=right, ordered=True,
        )
    audit = {
        "n_source_rows": int(len(raw)),
        "n_after_basic_preparation": int(len(prepared)),
        "n_eligible_before_dedup": int(len(eligible)),
        "n_overall": int(len(overall)),
        "n_by_dx": int(len(by_dx)),
        "dedup_sort_source": sort_source,
        "n_longitudinal_rows_removed_overall": int(removed_overall),
        "n_longitudinal_rows_removed_by_dx": int(removed_by_dx),
        "eligibility_requires_sex": bool(MATCH_FIG_S2_ELIGIBILITY),
    }
    return PanelSample(overall=overall, by_dx=by_dx, source=path.name), audit


def loess_bootstrap_ci(x, y, frac: float = LOESS_FRAC, n_boot: int = LOESS_N_BOOT,
                       alpha: float = ALPHA, n_grid: int = LOESS_GRID_N):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 5 or np.unique(x).size < 3:
        raise ValueError("Insufficient observations or unique x values for LOESS")

    x_grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    y_grid = np.interp(x_grid, smoothed[:, 0], smoothed[:, 1])

    curves = []
    n = len(x)
    for _ in range(n_boot):
        idx = RNG.choice(n, n, replace=True)
        xb, yb = x[idx], y[idx]
        if np.unique(xb).size < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resampled = lowess(yb, xb, frac=frac, return_sorted=True)
        if len(resampled) < 2:
            continue
        curves.append(np.interp(x_grid, resampled[:, 0], resampled[:, 1]))

    minimum_curves = max(20, int(0.5 * n_boot))
    if len(curves) < minimum_curves:
        raise RuntimeError(
            f"Only {len(curves)} valid bootstrap curves from {n_boot} replicates; "
            f"{minimum_curves} required"
        )

    curves = np.asarray(curves)
    low = np.percentile(curves, 100 * alpha / 2, axis=0)
    high = np.percentile(curves, 100 * (1 - alpha / 2), axis=0)
    return x_grid, y_grid, low, high, len(curves)


def plot_loess_panel(ax: plt.Axes, data: pd.DataFrame, x_col: str = "CL",
                     y_col: str = "zscore_2sd") -> None:
    for apoe in APOE_ORDER:
        group = data.loc[data["apoe_key"].eq(apoe)].dropna(subset=[x_col, y_col])
        if len(group) < 5:
            continue
        color = PALETTE_APOE[apoe]
        ax.scatter(group[x_col], group[y_col], color=color, alpha=0.18, s=16,
                   linewidths=0, zorder=2)
        try:
            x_grid, y_grid, low, high, _ = loess_bootstrap_ci(
                group[x_col].to_numpy(float), group[y_col].to_numpy(float)
            )
            ax.plot(x_grid, y_grid, color=color, lw=2, zorder=4)
            ax.fill_between(x_grid, low, high, color=color, alpha=0.20, zorder=3)
        except Exception as error:
            print(f"  [LOESS warning] {apoe}: {error}")


def compute_significance_by_bins(data: pd.DataFrame, bin_col: str,
                                 bin_labels: list[str], outcome_col: str,
                                 row_name: str, panel_name: str,
                                 groups: list[str],
                                 drawn_in_figure: bool = True):
    bin_records = []
    pair_records = []

    for position, label in enumerate(bin_labels):
        sub_bin = data.loc[bin_mask(data, bin_col, label)]
        counts = {}
        rows = []

        for apoe in groups:
            group = sub_bin.loc[sub_bin["apoe_key"].eq(apoe)].dropna(subset=[outcome_col])
            n = int(len(group))
            k = int(group[outcome_col].sum()) if n else 0
            counts[apoe] = {"n": n, "k": k}
            if n >= MIN_N_TEST:
                rows.append([k, n - k])

        global_p = np.nan
        global_chi2 = np.nan
        global_df = np.nan
        min_expected = np.nan
        if len(rows) >= 2:
            table = np.asarray(rows, dtype=int)
            if table[:, 0].sum() > 0 and table[:, 1].sum() > 0:
                try:
                    chi2, p, dof, expected = chi2_contingency(table, correction=False)
                    global_p = float(p)
                    global_chi2 = float(chi2)
                    global_df = int(dof)
                    min_expected = float(expected.min())
                except Exception:
                    pass

        bin_records.append({
            "row_name": row_name,
            "tested_genotypes": ",".join(groups),
            "panel": panel_name,
            "bin_position": float(position),
            "bin_label": label,
            "outcome_column": outcome_col,
            "global_chi2": global_chi2,
            "global_df": global_df,
            "global_p_raw": global_p,
            "global_min_expected": min_expected,
            "global_expected_counts_adequate": (
                bool(min_expected >= 5) if np.isfinite(min_expected) else np.nan
            ),
            "markers_rendered_in_figure": bool(drawn_in_figure),
            **{f"n_{a}": counts[a]["n"] for a in groups},
            **{f"k_{a}": counts[a]["k"] for a in groups},
        })

        for a1, a2 in combinations(groups, 2):
            n1, k1 = counts[a1]["n"], counts[a1]["k"]
            n2, k2 = counts[a2]["n"], counts[a2]["k"]
            p_raw = np.nan
            odds_ratio = np.nan
            if n1 >= MIN_N_TEST and n2 >= MIN_N_TEST:
                table = np.array([[k1, n1 - k1], [k2, n2 - k2]], dtype=int)
                try:
                    odds_ratio, p_raw = fisher_exact(table, alternative=FISHER_ALTERNATIVE)
                    odds_ratio = float(odds_ratio)
                    p_raw = float(p_raw)
                except Exception:
                    pass

            pair_records.append({
                "row_name": row_name,
                "tested_genotypes": ",".join(groups),
                "panel": panel_name,
                "bin_position": float(position),
                "bin_label": label,
                "outcome_column": outcome_col,
                "comparison": f"{a1}_vs_{a2}",
                "apoe_1": a1,
                "apoe_2": a2,
                "n1": n1, "k1": k1, "n2": n2, "k2": k2,
                "odds_ratio": odds_ratio,
                "p_raw": p_raw,
                "alternative": FISHER_ALTERNATIVE,
                "markers_rendered_in_figure": bool(drawn_in_figure),
            })

    for record, p_adj in zip(bin_records, holm_adjust([r["global_p_raw"] for r in bin_records])):
        record["global_p_holm"] = p_adj
        record["global_significant_holm"] = bool(np.isfinite(p_adj) and p_adj < ALPHA)

    for record, p_adj in zip(pair_records, holm_adjust([r["p_raw"] for r in pair_records])):
        record["p_holm"] = p_adj
        record["significant_holm"] = bool(np.isfinite(p_adj) and p_adj < ALPHA)

    summary = {}
    for record in bin_records:
        position = record["bin_position"]
        any_pair = any(
            pair["significant_holm"] for pair in pair_records
            if pair["bin_position"] == position
        )
        record["marker_criteria_met"] = bool(record["global_significant_holm"] and any_pair)
        record["marker_in_figure"] = bool(record["marker_criteria_met"] and drawn_in_figure)
        summary[position] = record.copy()

    return summary, bin_records, pair_records


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_SIZE["tick"])
    ax.grid(False)


def add_title_with_box(ax: plt.Axes, title: str) -> None:
    fig = ax.get_figure()
    ax_h_pts = ax.get_position().height * fig.get_size_inches()[1] * 72
    band_h = TITLE_BAND_PTS / ax_h_pts
    ax.add_patch(Rectangle(
        (0, 1.0), 1.0, band_h,
        transform=ax.transAxes,
        facecolor="#EBEBEB",
        edgecolor="black",
        linewidth=plt.rcParams.get("axes.linewidth", 0.8),
        clip_on=False,
        zorder=5,
    ))
    ax.text(
        0.5, 1.0 + band_h * 0.5, title,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=FONT_SIZE["title"],
        clip_on=False,
        zorder=6,
    )


def set_yticks_no_grid(ax: plt.Axes, ticks: list[float]) -> None:
    ax.set_yticks(ticks)
    ax.grid(False)


def add_row_label(fig: plt.Figure, y: float, text: str, start_x: float) -> None:
    prefix, remainder = text.split(" ", 1) if " " in text else (text, "")
    x_prefix = max(0.005, start_x + ROW_LABEL_X_SHIFT)
    fig.text(x_prefix, y, prefix, ha="left", va="bottom",
             fontsize=FONT_SIZE["title"] + 1, fontweight="bold")
    if remainder:
        fig.text(x_prefix + ROW_LABEL_TEXT_GAP, y, remainder, ha="left", va="bottom",
                 fontsize=FONT_SIZE["title"] + 1, fontweight="normal")


def group_positions(n_bins: int, n_groups: int):
    group_w = n_groups * BAR_WIDTH + (n_groups - 1) * BAR_SMALL_GAP
    x_centers = np.arange(n_bins) * (group_w + BAR_GROUP_GAP)
    offsets = (
        (np.arange(n_groups, dtype=float) - (n_groups - 1) / 2)
        * (BAR_WIDTH + BAR_SMALL_GAP)
    )
    return group_w, x_centers, offsets


def draw_rate_curve(ax: plt.Axes, data: pd.DataFrame, bin_col: str,
                    labels: list[str], positions: np.ndarray, outcome_col: str,
                    apoe: str, style: dict) -> None:
    color = PALETTE_APOE[apoe]
    group = data.loc[data["apoe_key"].eq(apoe)].dropna(subset=[bin_col, outcome_col])

    xs, ys = [], []
    for position, label in zip(positions, labels):
        subset = group.loc[bin_mask(group, bin_col, label)]
        if len(subset) < MIN_N:
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            xs.append(float(position))
            ys.append(float(subset[outcome_col].mean() * 100))

    ax.plot(
        np.asarray(xs, dtype=float), np.asarray(ys, dtype=float),
        color=color, lw=1.2,
        ls=style["linestyle"], marker=style["marker"], markersize=5,
        markerfacecolor=color if style["fill"] else "white",
        markeredgecolor=color, markeredgewidth=1.0, zorder=4,
    )


def plot_rate_panel(ax: plt.Axes, data: pd.DataFrame, significance_summary: dict,
                    bin_col: str, labels: list[str], positions: np.ndarray,
                    xlabel: str, ylabel: str, outcome_col: str,
                    groups: list[str], style: dict, title: str = "") -> None:
    for apoe in groups:
        draw_rate_curve(ax, data, bin_col, labels, positions, outcome_col, apoe, style)

    for position, info in significance_summary.items():
        if info.get("marker_in_figure", False):
            ax.scatter(
                position, 107,
                marker=style["marker"], color=style["color_marker"],
                s=30, zorder=8, clip_on=False, linewidths=1.0,
                edgecolors="#222222",
            )

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticks(positions)
    rotation = 40 if len(labels) > 8 else 0
    ax.set_xticklabels(
        labels,
        fontsize=FONT_SIZE["tick"] - (1.5 if len(labels) > 8 else 0.5),
        rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.set_ylim(-5, 115)
    set_yticks_no_grid(ax, [0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE["axis_label"])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE["axis_label"], labelpad=8)
    style_ax(ax)
    if title:
        add_title_with_box(ax, title)


def plot_binary_rate_by_bin(ax: plt.Axes, data: pd.DataFrame, bin_column: str,
                            bin_labels: list[str], xlabel: str,
                            significance_summary: Optional[dict] = None,
                            title: str = "") -> None:
    group_w, x_centers, offsets = group_positions(len(bin_labels), len(VISUAL_READ_APOE))

    for genotype_index, apoe in enumerate(VISUAL_READ_APOE):
        group = data.loc[data["apoe_key"].eq(apoe)].dropna(subset=[bin_column, "positive"])
        for bin_index, label in enumerate(bin_labels):
            subset = group.loc[bin_mask(group, bin_column, label)]
            total = int(len(subset))
            if total < MIN_BIN_N:
                continue
            successes = int(subset["positive"].sum())
            percent = 100.0 * successes / total
            low, high = wilson_interval(successes, total)
            x = x_centers[bin_index] + offsets[genotype_index]

            ax.bar(x, percent, width=BAR_WIDTH, color=PALETTE_APOE[apoe],
                   edgecolor="black", linewidth=0.5, zorder=3)
            if percent > 0:
                ax.errorbar(
                    x, percent,
                    yerr=[[max(0.0, percent - low)], [max(0.0, high - percent)]],
                    fmt="none", ecolor="#404040", elinewidth=1.0,
                    capsize=2.0, capthick=1.0, zorder=4,
                )

    y_top = 105.0
    if significance_summary and VISUAL_READ_SHOW_SIGNIFICANCE:
        y_top = 115.0
        for position, info in significance_summary.items():
            if info.get("marker_in_figure", False):
                ax.scatter(
                    x_centers[int(position)], 107,
                    marker="o", color="#222222", s=30, zorder=8,
                    clip_on=False, linewidths=1.0, edgecolors="#222222",
                )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(bin_labels, fontsize=FONT_SIZE["tick"] - 2.5)
    ax.set_xlim(x_centers[0] - group_w / 2 - 0.1, x_centers[-1] + group_w / 2 + 0.1)
    ax.set_ylim(0, y_top)
    set_yticks_no_grid(ax, [0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE["axis_label"])
    ax.set_ylabel("Positive visual read (%)", fontsize=FONT_SIZE["axis_label"], labelpad=8)
    style_ax(ax)
    if title:
        add_title_with_box(ax, title)


ROW_PLAN = [
    {"key": "loess", "kind": "loess", "section": 0},
    {"key": "ptau_rate", "kind": "rate", "section": 1},
    {"key": "visual_read_cl", "kind": "bar", "section": 2},
    {"key": "visual_read_ptau", "kind": "bar", "section": 2},
]


@dataclass
class Geometry:
    bottoms: list[float]
    heights: list[float]
    start_x: float
    total_w: float
    fig_height_in: float

    def inches(self, value: float) -> float:
        return value / self.fig_height_in


def compute_row_geometry() -> Geometry:
    heights_in = [ROW_H_IN[row["kind"]] for row in ROW_PLAN]
    gaps_in = [0.0]
    for index in range(1, len(ROW_PLAN)):
        previous_section = ROW_PLAN[index - 1]["section"]
        current_section = ROW_PLAN[index]["section"]
        if current_section == previous_section:
            gaps_in.append(GAP_WITHIN_SECTION_IN)
        elif current_section == 1:
            gaps_in.append(GAP_BETWEEN_A_AND_B_IN)
        else:
            gaps_in.append(GAP_BETWEEN_B_AND_C_IN)

    bottoms_in = [0.0] * len(ROW_PLAN)
    bottoms_in[-1] = BOTTOM_MARGIN_IN
    for index in range(len(ROW_PLAN) - 2, -1, -1):
        bottoms_in[index] = bottoms_in[index + 1] + heights_in[index + 1] + gaps_in[index + 1]

    fig_height_in = bottoms_in[0] + heights_in[0] + TOP_MARGIN_IN

    total_w = 4 * PANEL_W + 3 * COL_GAP
    if total_w > 1.0:
        raise ValueError("The panel block exceeds the figure width.")

    return Geometry(
        bottoms=[value / fig_height_in for value in bottoms_in],
        heights=[value / fig_height_in for value in heights_in],
        start_x=(1.0 - total_w) / 2,
        total_w=total_w,
        fig_height_in=fig_height_in,
    )


def place_legend(fig: plt.Figure, handles: list, y: float, ncol: int) -> None:
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=FONT_SIZE["legend"],
        handlelength=2.4,
        columnspacing=1.6,
    )


def build_figure(ptau_sample: PanelSample, visual_read_cl: PanelSample,
                 visual_read_ptau: PanelSample, significance: dict) -> None:
    print("\nBuilding figure...")
    geometry = compute_row_geometry()
    bottoms, heights, start_x = geometry.bottoms, geometry.heights, geometry.start_x
    print(f"  Canvas: {FIG_W} x {geometry.fig_height_in:.2f} inches")
    fig = plt.figure(
        figsize=(FIG_W, geometry.fig_height_in), dpi=300, facecolor="white"
    )

    for row_index, row in enumerate(ROW_PLAN):
        for column_index, panel in enumerate(PANEL_ORDER):
            panel_x = start_x + column_index * (PANEL_W + COL_GAP)
            ax = fig.add_axes([panel_x, bottoms[row_index], PANEL_W, heights[row_index]])
            ax.set_facecolor("white")
            style_ax(ax)

            if row["key"] == "loess":
                data = ptau_sample.panel(panel)
                plot_loess_panel(ax, data)
                ax.set_xlim(0, 110)
                ax.set_ylim(0, 4)
                ax.set_xlabel("Aβ burden (CL)", fontsize=FONT_SIZE["axis_label"])
                ax.set_ylabel("Plasma p-tau217 (z)", fontsize=FONT_SIZE["axis_label"],
                              labelpad=8)
                ax.grid(False)
                add_title_with_box(ax, panel)

            elif row["key"] == "ptau_rate":
                plot_rate_panel(
                    ax, ptau_sample.panel(panel), significance["ptau_cl"][panel],
                    "CL10_bin", CL10_LABELS, CL10_POS,
                    xlabel="Aβ burden (CL)",
                    ylabel="Plasma p-tau217 positivity (%)",
                    outcome_col=PTAU_POS_COL,
                    groups=APOE_ORDER,
                    style=PTAU_SERIES_STYLE,
                    title=panel,
                )

            elif row["key"] == "visual_read_cl":
                plot_binary_rate_by_bin(
                    ax, visual_read_cl.panel(panel), "CL_bin", CL_BIN_LABELS,
                    "Aβ burden (CL)",
                    significance_summary=significance["visual_read_cl"][panel],
                    title=panel,
                )

            else:
                plot_binary_rate_by_bin(
                    ax, visual_read_ptau.panel(panel), "zscore_bin", ZSCORE_BIN_LABELS,
                    "Plasma p-tau217 (z)",
                    significance_summary=significance["visual_read_ptau"][panel],
                    title=panel,
                )

    first_row_of_section: dict[int, int] = {}
    last_row_of_section: dict[int, int] = {}
    for index, row in enumerate(ROW_PLAN):
        first_row_of_section.setdefault(row["section"], index)
        last_row_of_section[row["section"]] = index

    for section, row_index in first_row_of_section.items():
        add_row_label(
            fig,
            bottoms[row_index] + heights[row_index]
            + geometry.inches(ROW_LABEL_Y_OFFSET_IN),
            SECTION_LABELS[section],
            start_x,
        )

    line_handles = [
        Line2D([0], [0], color=PALETTE_APOE[apoe], lw=2, label=APOE_LABELS[apoe])
        for apoe in APOE_ORDER
    ]
    bar_handles = [
        Patch(facecolor=PALETTE_APOE[apoe], edgecolor="black", linewidth=0.5,
              label=APOE_LABELS[apoe])
        for apoe in VISUAL_READ_APOE
    ]
    bar_handles.append(
        Line2D([0], [0], color="#404040", lw=1.0, marker="_", markersize=10,
               markeredgewidth=1.0, label="Wilson 95% CI")
    )

    fig.canvas.draw()
    for section, handles in [
        (0, line_handles), (1, line_handles), (2, bar_handles)
    ]:
        anchor_row = (
            last_row_of_section[section] if section == 2
            else first_row_of_section[section]
        )
        place_legend(
            fig, handles,
            bottoms[anchor_row] + geometry.inches(LEGEND_Y_OFFSET_IN[section]),
            len(handles),
        )

    fig.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {FIGURE_FILE}")


def save_dataframe(records, path: Path) -> None:
    frame = pd.DataFrame(records)
    frame.to_csv(path, index=False)
    print(f"Saved: {path} ({len(frame):,} rows)")


def export_selection_audit(ptau_audit: dict, cl_audit: dict, ptau_vr_audit: dict) -> None:
    records = []

    def add(section, analysis_sample, source, dedup_subset, visit_rule, audit,
            n_after_dedup, **extra):
        records.append({
            "figure_section": section,
            "section_label": SECTION_LABELS[section],
            "analysis_sample": analysis_sample,
            "source_file": source,
            "n_source_rows": audit["n_source_rows"],
            "n_after_basic_preparation": audit["n_after_basic_preparation"],
            "n_eligible_before_dedup": audit["n_eligible_before_dedup"],
            "n_after_dedup": n_after_dedup,
            "dedup_subset": dedup_subset,
            "visit_rule": visit_rule,
            "max_abs_days_ptau_to_pet": MAX_DAYS_PTAU_PET,
            "genotypes_in_sample": ",".join(
                APOE_ORDER if section in (0, 1) else VISUAL_READ_APOE
            ),
            "ptau_positive_threshold_z": (
                PTAU_POS_THRESHOLD if section in (0, 1) else np.nan
            ),
            "min_cell_n_to_draw": MIN_N if section in (0, 1) else MIN_BIN_N,
            **extra,
        })

    ptau_rule = f"{PTAU_VISIT_KEEP} eligible visit by AGE_at_visit"
    for section in (0, 1):
        add(section, "pTau217_overall_baseline", PTAU_FILE.name, "BIDS_SUBJECT",
            ptau_rule, ptau_audit, ptau_audit["n_overall"])
        add(section, "pTau217_within_DX_baseline", PTAU_FILE.name,
            "BIDS_SUBJECT + DX_LABEL", ptau_rule, ptau_audit, ptau_audit["n_by_dx"])

    for audit, source, name in [
        (cl_audit, VISUAL_READ_CL_FILE.name, "VisualRead_CL"),
        (ptau_vr_audit, VISUAL_READ_PTAU_FILE.name, "VisualRead_pTau217"),
    ]:
        rule = f"earliest eligible visit by {audit['dedup_sort_source']}"
        add(2, f"{name}_overall", source, "BIDS_SUBJECT", rule, audit,
            audit["n_overall"],
            eligibility_requires_sex=audit["eligibility_requires_sex"],
            n_longitudinal_rows_removed=audit["n_longitudinal_rows_removed_overall"])
        add(2, f"{name}_within_DX", source, "BIDS_SUBJECT + DX_LABEL", rule, audit,
            audit["n_by_dx"],
            eligibility_requires_sex=audit["eligibility_requires_sex"],
            n_longitudinal_rows_removed=audit["n_longitudinal_rows_removed_by_dx"])

    save_dataframe(records, SELECTION_AUDIT_FILE)


def export_loess_anchors(ptau_sample: PanelSample) -> None:
    records = []
    for panel in PANEL_ORDER:
        data = ptau_sample.panel(panel)
        for apoe in APOE_ORDER:
            group = data.loc[data["apoe_key"].eq(apoe)].dropna(subset=["CL", "zscore_2sd"])
            if len(group) < 5 or group["CL"].nunique() < 3:
                continue
            smoothed = lowess(
                group["zscore_2sd"].to_numpy(float), group["CL"].to_numpy(float),
                frac=LOESS_FRAC, return_sorted=True,
            )
            for anchor in CL_ANCHORS:
                in_range = smoothed[:, 0].min() <= anchor <= smoothed[:, 0].max()
                records.append({
                    "figure_section": 0,
                    "section_label": SECTION_LABELS[0],
                    "panel": panel,
                    "panel_sample": PANEL_SAMPLE_RULE[panel],
                    "APOE": apoe,
                    "CL_anchor": float(anchor),
                    "loess_estimated_pTau217_z": (
                        float(np.interp(anchor, smoothed[:, 0], smoothed[:, 1]))
                        if in_range else np.nan
                    ),
                    "in_observed_CL_range": bool(in_range),
                    "n_subjects": int(len(group)),
                    "LOESS_fraction": LOESS_FRAC,
                    "LOESS_bootstrap_replicates": LOESS_N_BOOT,
                    "data_source": ptau_sample.source,
                    "max_abs_days_ptau_to_pet": MAX_DAYS_PTAU_PET,
                })
    save_dataframe(records, LOESS_FILE)


def export_bin_proportions(visual_read_cl: PanelSample,
                           visual_read_ptau: PanelSample) -> None:
    records = []
    for sample, bin_column, bin_labels, x_variable, x_label in [
        (visual_read_cl, "CL_bin", CL_BIN_LABELS, "Abeta_burden_CL", "Aβ burden (CL)"),
        (visual_read_ptau, "zscore_bin", ZSCORE_BIN_LABELS, "pTau217_zscore",
         "Plasma p-tau217 (z)"),
    ]:
        for panel in PANEL_ORDER:
            data = sample.panel(panel)
            for apoe in [*VISUAL_READ_APOE, "All"]:
                group = data if apoe == "All" else data.loc[data["apoe_key"].eq(apoe)]
                group = group.dropna(subset=[bin_column, "positive"])
                for label in bin_labels:
                    subset = group.loc[bin_mask(group, bin_column, label)]
                    total = int(len(subset))
                    successes = int(subset["positive"].sum())
                    low, high = wilson_interval(successes, total)
                    records.append({
                        "figure_section": 2,
                        "section_label": SECTION_LABELS[2],
                        "panel": panel,
                        "panel_sample": PANEL_SAMPLE_RULE[panel],
                        "x_variable": x_variable,
                        "x_axis_label": x_label,
                        "x_bin_label": label,
                        "APOE": apoe,
                        "e4_dose": E4_DOSE.get(apoe, np.nan),
                        "n_subjects": total,
                        "n_positive": successes,
                        "pct_positive": np.nan if total == 0 else 100.0 * successes / total,
                        "wilson_ci_lower": low,
                        "wilson_ci_upper": high,
                        "below_min_n": bool(total < MIN_BIN_N),
                        "drawn_in_figure": bool(total >= MIN_BIN_N and apoe != "All"),
                        "min_bin_n": MIN_BIN_N,
                        "outcome_column": VISUAL_READ_OUTCOME_COL,
                        "data_source": sample.source,
                    })
    save_dataframe(records, BIN_PROPORTIONS_FILE)


def export_descriptives(ptau_sample: PanelSample, visual_read_cl: PanelSample,
                        visual_read_ptau: PanelSample) -> None:
    records = []

    specs = [
        (0, ptau_sample, "AGE_at_visit", "CL", "zscore_2sd", None, APOE_ORDER),
        (1, ptau_sample, "AGE_at_visit", "CL", "zscore_2sd", PTAU_POS_COL, APOE_ORDER),
        (2, visual_read_cl, "AGE", "CL", None, "positive", VISUAL_READ_APOE),
        (2, visual_read_ptau, "AGE", None, "zscore_2sd", "positive", VISUAL_READ_APOE),
    ]

    for section, sample, age_col, cl_col, z_col, outcome_col, genotypes in specs:
        for panel in PANEL_ORDER:
            data = sample.panel(panel)
            for apoe in [*genotypes, "All"]:
                subset = data if apoe == "All" else data.loc[data["apoe_key"].eq(apoe)]
                record = {
                    "figure_section": section,
                    "section_label": SECTION_LABELS[section],
                    "panel": panel,
                    "panel_sample": PANEL_SAMPLE_RULE[panel],
                    "APOE": apoe,
                    "e4_dose": E4_DOSE.get(apoe, np.nan),
                    "data_source": sample.source,
                    "n_rows": int(len(subset)),
                    "n_unique_subjects": int(subset["BIDS_SUBJECT"].nunique()),
                    "max_abs_days_ptau_to_pet": MAX_DAYS_PTAU_PET,
                    "ptau_positive_threshold_z": PTAU_POS_THRESHOLD,
                    "drawn_in_figure": bool(apoe != "All"),
                }
                record.update(numeric_summary(subset[age_col], "age"))
                if cl_col and cl_col in subset.columns:
                    record.update(numeric_summary(subset[cl_col], "CL"))
                if z_col and z_col in subset.columns:
                    record.update(numeric_summary(subset[z_col], "pTau217_zscore"))
                if outcome_col and outcome_col in subset.columns:
                    values = pd.to_numeric(subset[outcome_col], errors="coerce").dropna()
                    total = int(len(values))
                    successes = int(values.sum()) if total else 0
                    low, high = wilson_interval(successes, total)
                    record.update({
                        "outcome_column": outcome_col,
                        "n_outcome_available": total,
                        "n_outcome_positive": successes,
                        "pct_outcome_positive": (
                            100.0 * successes / total if total else np.nan
                        ),
                        "wilson_ci_lower": low,
                        "wilson_ci_upper": high,
                    })
                records.append(record)
    save_dataframe(records, DESCRIPTIVES_FILE)


def export_subplot_and_detailed(ptau_sample: PanelSample,
                                visual_read_cl: PanelSample,
                                visual_read_ptau: PanelSample,
                                significance: dict) -> None:
    summary_rows: list[dict] = []
    detailed_rows: list[dict] = []

    for panel in PANEL_ORDER:
        data = ptau_sample.panel(panel)
        for apoe in [*APOE_ORDER, "All"]:
            subset = data if apoe == "All" else data.loc[data["apoe_key"].eq(apoe)]
            record = {
                "figure_section": 0,
                "section_label": SECTION_LABELS[0],
                "panel": panel,
                "panel_sample": PANEL_SAMPLE_RULE[panel],
                "APOE": apoe,
                "plot_type": "scatter_LOESS_bootstrap_CI",
                "data_source": ptau_sample.source,
                "max_abs_days_ptau_to_pet": MAX_DAYS_PTAU_PET,
                "LOESS_fraction": LOESS_FRAC,
                "LOESS_bootstrap_replicates": LOESS_N_BOOT,
                "ptau_positive_threshold_z": PTAU_POS_THRESHOLD,
            }
            record.update(numeric_summary(subset["CL"], "CL"))
            record.update(numeric_summary(subset["zscore_2sd"], "pTau217_zscore"))
            record.update(association_summary(subset, "CL", "zscore_2sd"))
            summary_rows.append(record)

            for anchor in CL_ANCHORS:
                window = subset.loc[
                    subset["CL"].between(anchor - 12.5, anchor + 12.5, inclusive="both")
                ]
                detailed_rows.append({
                    "figure_section": 0,
                    "section_label": SECTION_LABELS[0],
                    "panel": panel,
                    "panel_sample": PANEL_SAMPLE_RULE[panel],
                    "APOE": apoe,
                    "measurement": "pTau217_zscore",
                    "CL_anchor": float(anchor),
                    "CL_window": f"{anchor - 12.5:g} to {anchor + 12.5:g}",
                    "n_subjects": int(len(window)),
                    "mean_pTau217_zscore": _safe_float(window["zscore_2sd"].mean()),
                    "sd_pTau217_zscore": _safe_float(window["zscore_2sd"].std(ddof=1)),
                    "n_pTau217_positive": int(
                        (window["zscore_2sd"] >= PTAU_POS_THRESHOLD).sum()
                    ),
                    "pct_pTau217_positive": (
                        float((window["zscore_2sd"] >= PTAU_POS_THRESHOLD).mean() * 100)
                        if len(window) else np.nan
                    ),
                    "ptau_positive_threshold_z": PTAU_POS_THRESHOLD,
                    "data_source": ptau_sample.source,
                    "max_abs_days_ptau_to_pet": MAX_DAYS_PTAU_PET,
                })

    binned_specs = [
        (1, ptau_sample, "CL10_bin", CL10_LABELS, CL10_POS, PTAU_POS_COL,
         significance["ptau_cl"], APOE_ORDER, "pTau217 positivity",
         "Abeta_burden_CL", MIN_N, True),
        (2, visual_read_cl, "CL_bin", CL_BIN_LABELS,
         np.arange(len(CL_BIN_LABELS), dtype=float), "positive",
         significance["visual_read_cl"], VISUAL_READ_APOE,
         "Visual read positivity", "Abeta_burden_CL", MIN_BIN_N,
         VISUAL_READ_SHOW_SIGNIFICANCE),
        (2, visual_read_ptau, "zscore_bin", ZSCORE_BIN_LABELS,
         np.arange(len(ZSCORE_BIN_LABELS), dtype=float), "positive",
         significance["visual_read_ptau"], VISUAL_READ_APOE,
         "Visual read positivity", "pTau217_zscore", MIN_BIN_N,
         VISUAL_READ_SHOW_SIGNIFICANCE),
    ]

    for (section, sample, bin_col, labels, positions, outcome_col,
         significance_lookup, genotypes, measurement, x_variable, min_cell_n,
         markers_drawn) in binned_specs:
        for panel in PANEL_ORDER:
            data = sample.panel(panel)
            panel_significance = significance_lookup[panel]
            for position, label in zip(positions, labels):
                sub_bin = data.loc[bin_mask(data, bin_col, label)]
                info = panel_significance.get(float(position), {})
                for apoe in [*genotypes, "All"]:
                    subset = sub_bin if apoe == "All" else sub_bin.loc[
                        sub_bin["apoe_key"].eq(apoe)
                    ]
                    values = pd.to_numeric(subset[outcome_col], errors="coerce").dropna()
                    total = int(len(values))
                    successes = int(values.sum()) if total else 0
                    low, high = wilson_interval(successes, total)
                    record = {
                        "figure_section": section,
                        "section_label": SECTION_LABELS[section],
                        "panel": panel,
                        "panel_sample": PANEL_SAMPLE_RULE[panel],
                        "APOE": apoe,
                        "e4_dose": E4_DOSE.get(apoe, np.nan),
                        "measurement": measurement,
                        "outcome_column": outcome_col,
                        "x_variable": x_variable,
                        "bin_variable": bin_col,
                        "bin_label": label,
                        "bin_position": float(position),
                        "n_subjects": total,
                        "n_positive": successes,
                        "pct_positive": 100.0 * successes / total if total else np.nan,
                        "wilson_ci_lower": low,
                        "wilson_ci_upper": high,
                        "min_cell_n_to_draw": min_cell_n,
                        "below_min_n": bool(total < min_cell_n),
                        "plotted_in_figure": bool(total >= min_cell_n and apoe != "All"),
                        "tested_genotypes": ",".join(genotypes),
                        "global_p_raw": info.get("global_p_raw", np.nan),
                        "global_p_holm": info.get("global_p_holm", np.nan),
                        "global_min_expected": info.get("global_min_expected", np.nan),
                        "marker_criteria_met": info.get("marker_criteria_met", False),
                        "significance_marker_in_figure": info.get("marker_in_figure", False),
                        "significance_markers_rendered_in_this_section": bool(markers_drawn),
                        "data_source": sample.source,
                        "max_abs_days_ptau_to_pet": MAX_DAYS_PTAU_PET,
                        "ptau_positive_threshold_z": (
                            PTAU_POS_THRESHOLD if outcome_col == PTAU_POS_COL else np.nan
                        ),
                    }
                    summary_rows.append(record.copy())
                    detailed_rows.append(record)

    save_dataframe(summary_rows, SUBPLOT_SUMMARY_FILE)
    save_dataframe(detailed_rows, DETAILED_FILE)


def compute_all_significance(ptau_sample: PanelSample, visual_read_cl: PanelSample,
                             visual_read_ptau: PanelSample) -> tuple[dict, list[dict]]:
    print("\nComputing multiplicity-corrected significance tests...")
    significance = {"ptau_cl": {}, "visual_read_cl": {}, "visual_read_ptau": {}}
    global_tests: list[dict] = []
    pairwise_tests: list[dict] = []

    specs = [
        ("ptau_cl", ptau_sample, "CL10_bin", CL10_LABELS, PTAU_POS_COL,
         "pTau217 positivity by Aβ burden", APOE_ORDER, True),
        ("visual_read_cl", visual_read_cl, "CL_bin", CL_BIN_LABELS, "positive",
         "Visual read positivity by Aβ burden", VISUAL_READ_APOE,
         VISUAL_READ_SHOW_SIGNIFICANCE),
        ("visual_read_ptau", visual_read_ptau, "zscore_bin", ZSCORE_BIN_LABELS,
         "positive", "Visual read positivity by plasma p-tau217", VISUAL_READ_APOE,
         VISUAL_READ_SHOW_SIGNIFICANCE),
    ]

    for key, sample, bin_col, labels, outcome_col, row_name, groups, drawn in specs:
        for panel in PANEL_ORDER:
            summary, bins, pairs = compute_significance_by_bins(
                sample.panel(panel), bin_col, labels, outcome_col,
                row_name=row_name, panel_name=panel, groups=groups,
                drawn_in_figure=drawn,
            )
            significance[key][panel] = summary
            global_tests.extend(bins)
            pairwise_tests.extend(pairs)

    records = [{"test_level": "global", **record} for record in global_tests]
    records += [{"test_level": "pairwise", **record} for record in pairwise_tests]
    return significance, records


def main() -> None:
    validate_configuration()

    print("Loading and preparing data...")
    print(f"  p-tau/PET matching window: ±{MAX_DAYS_PTAU_PET} days")
    print(f"  p-tau217 positivity threshold: z >= {PTAU_POS_THRESHOLD}")
    print(f"  Genotypes in sections a and b: {', '.join(APOE_ORDER)}")
    print(f"  Genotypes in section c:        {', '.join(VISUAL_READ_APOE)}")
    print(f"  LOESS: frac = {LOESS_FRAC}, {LOESS_N_BOOT} bootstrap replicates")

    ptau_sample, ptau_audit = build_ptau_sample()
    visual_read_cl, cl_audit = build_visual_read_sample(
        VISUAL_READ_CL_FILE, include_ptau=False, bin_column="CL_bin",
        bin_source="CL", bin_edges=CL_BIN_EDGES, bin_labels=CL_BIN_LABELS, right=True,
    )
    visual_read_ptau, ptau_vr_audit = build_visual_read_sample(
        VISUAL_READ_PTAU_FILE, include_ptau=True, bin_column="zscore_bin",
        bin_source="zscore_2sd", bin_edges=ZSCORE_BIN_EDGES,
        bin_labels=ZSCORE_BIN_LABELS, right=False,
    )

    print(f"\n  Sections a and b: n = {len(ptau_sample.overall):,} (Overall)")
    print(f"  Section c upper:  n = {len(visual_read_cl.overall):,} (Overall)")
    print(f"  Section c lower:  n = {len(visual_read_ptau.overall):,} (Overall)")

    significance, significance_records = compute_all_significance(
        ptau_sample, visual_read_cl, visual_read_ptau
    )

    build_figure(ptau_sample, visual_read_cl, visual_read_ptau, significance)

    print("\nCreating quantitative exports...")
    export_selection_audit(ptau_audit, cl_audit, ptau_vr_audit)
    save_dataframe(significance_records, SIGNIFICANCE_FILE)
    export_loess_anchors(ptau_sample)
    export_bin_proportions(visual_read_cl, visual_read_ptau)
    export_descriptives(ptau_sample, visual_read_cl, visual_read_ptau)
    export_subplot_and_detailed(ptau_sample, visual_read_cl, visual_read_ptau, significance)

    print("\nSaved outputs:")
    for path in ALL_OUTPUT_FILES:
        print(f"  {path}")
    print("Done.")


if __name__ == "__main__":
    main()
