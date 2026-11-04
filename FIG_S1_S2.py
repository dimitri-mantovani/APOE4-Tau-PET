"""
Supplementary Figures S1 and S2. Regional tau PET positivity (Braak I–II,
III–IV and V–VI) by Aβ burden, plasma p-tau217, clinical status and APOE
haplotype.

The tau tracer is selected with TAU_TRACER ("FTP" for [18F]flortaucipir, "MK"
for [18F]MK-6240); each run produces the figure for one tracer. The script uses
APOE 2/3, 3/3, 3/4 and 4/4 participants, excludes co-pathology (COPAT = 1),
requires p-tau217 within 365 days of the PET scan, and keeps the earliest
eligible visit per participant (Overall column) and per participant and
diagnosis (CU, MCI and Dementia columns). Centiloid and tau PET values come
from input datasets in which the amyloid and tau PET scans were already matched
in time, so no additional amyloid-to-tau PET interval filter is applied here.
For each Braak region it plots the percentage of tau-positive participants
across Centiloid bins and p-tau217 z-score bins (cells with at least 5
participants). Markers indicate bins in which haplotypes differ (omnibus chi-
square or Monte Carlo exact test and pairwise Fisher exact tests, Holm-
corrected).

Inputs:  DIMI_DATASET_07-09-2026.csv, DIMI_with_ptau217_07-09-2026.csv
Outputs: FIG_S2_<tracer>.png and CSV tables with bin-level positivity, region
         summaries, test results, sample summaries and sample selection.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, fisher_exact, random_table


BASE_FILE = Path("DIMI_DATASET_07-09-2026.csv")
PTAU_FILE = Path("DIMI_with_ptau217_07-09-2026.csv")

TAU_TRACER = "MK"

SUBJECT_ID_COL = "BIDS_SUBJECT"
MAX_DAYS_PTAU_PET = 365
VISIT_SELECTION = "earliest"

MIN_N = 5
MIN_N_TEST = 5
ALPHA = 0.05
GLOBAL_MONTE_CARLO_RESAMPLES = 10_000
RNG_GLOBAL = np.random.default_rng(42)


TRACER_CONFIGS: dict[str, dict[str, Any]] = {
    "MK": {
        "key": "MK",
        "name": "[$^{18}$F]MK-6240",
        "short_name": "[$^{18}$F]MK-6240",
        "output_tag": "MK",
        "column_sets": [
            {
                "Braak 1–2": "MK_POS_BRAAK12_COH",
                "Braak 3–4": "MK_POS_BRAAK34_COH",
                "Braak 5–6": "MK_POS_BRAAK56_COH",
            },
            {
                "Braak 1–2": "MK6240_POS_BRAAK12_COH",
                "Braak 3–4": "MK6240_POS_BRAAK34_COH",
                "Braak 5–6": "MK6240_POS_BRAAK56_COH",
            },
        ],
    },
    "FTP": {
        "key": "FTP",
        "name": "[$^{18}$F]flortaucipir",
        "short_name": "[$^{18}$F]flortaucipir",
        "output_tag": "FTP",
        "column_sets": [
            {
                "Braak 1–2": "POS_BRAAK12_COH",
                "Braak 3–4": "POS_BRAAK34_COH",
                "Braak 5–6": "POS_BRAAK56_COH",
            },
            {
                "Braak 1–2": "FTP_POS_BRAAK12_COH",
                "Braak 3–4": "FTP_POS_BRAAK34_COH",
                "Braak 5–6": "FTP_POS_BRAAK56_COH",
            },
        ],
    },
}


def _available_tracer_options(
    base: pd.DataFrame,
    ptau: pd.DataFrame,
) -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    for key, config in TRACER_CONFIGS.items():
        for columns in config["column_sets"]:
            required = list(columns.values())
            if all(c in base.columns and c in ptau.columns for c in required):
                complete_base = int(base[required].notna().all(axis=1).sum())
                complete_ptau = int(ptau[required].notna().all(axis=1).sum())
                available.append(
                    {
                        **{k: v for k, v in config.items() if k != "column_sets"},
                        "columns": columns,
                        "score": complete_base + complete_ptau,
                        "complete_base": complete_base,
                        "complete_ptau": complete_ptau,
                    }
                )
                break
    return available


def resolve_tracer(base: pd.DataFrame, ptau: pd.DataFrame) -> dict[str, Any]:
    requested = TAU_TRACER.strip().upper()
    if requested not in {"MK", "FTP", "AUTO"}:
        raise ValueError("TAU_TRACER must be 'MK', 'FTP', or 'AUTO'.")

    options = _available_tracer_options(base, ptau)
    if not options:
        expected = {
            key: [list(s.values()) for s in cfg["column_sets"]]
            for key, cfg in TRACER_CONFIGS.items()
        }
        raise KeyError(
            "No complete MK or FTP Braak positivity column family was found "
            f"in both input files. Expected one of: {expected}"
        )

    if requested != "AUTO":
        matching = [o for o in options if o["key"] == requested]
        if not matching:
            raise KeyError(
                f"TAU_TRACER='{requested}', but its three regional positivity "
                "columns are not present in both input files."
            )
        return matching[0]

    return sorted(
        options,
        key=lambda o: (o["score"], o["key"] == "MK"),
        reverse=True,
    )[0]


PTAU_ZSCORE_CANDIDATES = [
    "zscore_2sd",
    "PTAU_ZSCORE",
    "pTau217_zscore",
    "ptau217_zscore",
]
PTAU_DAYS_CANDIDATES = [
    "days_diff",
    "DAYS_DIFF",
    "days_pTau_to_PET",
    "DAYS_PTAU_TO_PET",
]
PTAU_ABS_YEARS_CANDIDATES = [
    "ABS_AGE_DIFF_YEARS",
    "abs_age_diff_years",
]
PTAU_SIGNED_YEARS_CANDIDATES = [
    "AGE_DIFF_YEARS",
    "age_diff_years",
]


FONT_SIZE = {
    "panel_label": 16,
    "title": 14,
    "axis_label": 14,
    "tick": 12,
    "legend": 14,
    "annot": 9,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
        "axes.titleweight": "regular",
        "font.size": FONT_SIZE["tick"],
        "axes.labelsize": FONT_SIZE["axis_label"],
        "xtick.labelsize": FONT_SIZE["tick"],
        "ytick.labelsize": FONT_SIZE["tick"],
    }
)

palette_apoe = {
    "44": "#C7353E",
    "34": "#E0B04A",
    "33": "#389F42",
    "23": "#2E86C1",
}
apoe_order = ["23", "33", "34", "44"]
SIGNIFICANCE_APOE_ORDER = ["44", "34", "33", "23"]
apoe_labels = {
    "44": r"$\it{APOE}$4/4",
    "34": r"$\it{APOE}$3/4",
    "33": r"$\it{APOE}$3/3",
    "23": r"$\it{APOE}$2/3",
}
DX_ORDER = ["Overall", "CU", "MCI", "Dementia"]
DX_MAP = {1: "CU", 2: "MCI", 3: "Dementia"}
APOE_ALLOWED = set(apoe_order)

CL_EDGES = [-np.inf, 25, 50, 75, 100, np.inf]
CL_BIN_LABELS = ["<25", "25–50", "50–75", "75–100", ">100"]
CL_BIN_X = np.arange(1, len(CL_BIN_LABELS) + 1, dtype=float)

ZSCORE_EDGES = [-np.inf, 0, 1.5, 3.0, 4.5, np.inf]
ZSCORE_LABELS = ["<0", "0–<1.5", "1.5–<3.0", "3.0–<4.5", "≥4.5"]
ZSCORE_BIN_X = np.arange(1, len(ZSCORE_LABELS) + 1, dtype=float)

LW_BIN = 1.15
MARKER_SIZE = 22
YLIM_BIN = (-5, 115)
Y_SIG = 107

PANEL_W = 0.188
COL_GAP = 0.048
ROW_H = 0.092
INTERNAL_ROW_GAP = 0.06
SECTION_GAP = 0.10
BOTTOM_MARGIN = 0.135
SECTION_LABEL_Y_OFFSET = 0.030
SECTION_LABEL_X_SHIFT = -0.062
SECTION_LABEL_TEXT_GAP = 0.023
LEGEND_Y = 0.07


def tracer_metadata_columns(tracer: dict[str, Any]) -> dict[str, Any]:
    return {
        "tau_tracer_key": tracer["key"],
        "tau_tracer_name": tracer["name"],
        "tau_tracer_short_name": tracer["short_name"],
        "tau_output_tag": tracer["output_tag"],
    }


def normalize_apoe_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    digits = re.findall(r"[234]", text)
    if len(digits) < 2:
        return None
    pair = "".join(sorted(digits[:2]))
    return pair if pair in APOE_ALLOWED else None


def e4_dose(apoe: str) -> int:
    return {"44": 2, "34": 1, "33": 0, "23": 0}[apoe]


def require_columns(df: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{source_name} is missing required columns: {missing}")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def standardize_ptau_columns(
    df: pd.DataFrame,
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    work = df.copy()
    z_col = first_existing_column(work, PTAU_ZSCORE_CANDIDATES)
    if z_col is None:
        raise KeyError(
            f"{source_name} has no recognized p-tau z-score column. "
            f"Tried: {PTAU_ZSCORE_CANDIDATES}"
        )
    work["PTAU_ZSCORE_STD"] = pd.to_numeric(work[z_col], errors="coerce")

    days_col = first_existing_column(work, PTAU_DAYS_CANDIDATES)
    abs_years_col = first_existing_column(work, PTAU_ABS_YEARS_CANDIDATES)
    signed_years_col = first_existing_column(work, PTAU_SIGNED_YEARS_CANDIDATES)

    if days_col is not None:
        work["ABS_DAYS_DIFF_STD"] = pd.to_numeric(
            work[days_col], errors="coerce"
        ).abs()
        interval_source = days_col
    elif abs_years_col is not None:
        work["ABS_DAYS_DIFF_STD"] = (
            pd.to_numeric(work[abs_years_col], errors="coerce").abs() * 365.25
        )
        interval_source = f"{abs_years_col} × 365.25"
    elif signed_years_col is not None:
        work["ABS_DAYS_DIFF_STD"] = (
            pd.to_numeric(work[signed_years_col], errors="coerce").abs() * 365.25
        )
        interval_source = f"abs({signed_years_col}) × 365.25"
    elif "PTAU_AGE" in work.columns and "AGE" in work.columns:
        work["ABS_DAYS_DIFF_STD"] = (
            (
                pd.to_numeric(work["PTAU_AGE"], errors="coerce")
                - pd.to_numeric(work["AGE"], errors="coerce")
            ).abs()
            * 365.25
        )
        interval_source = "abs(PTAU_AGE - AGE) × 365.25"
    else:
        raise KeyError(
            f"{source_name} has no recognized p-tau/PET interval column. "
            f"Tried day columns {PTAU_DAYS_CANDIDATES}, year columns "
            f"{PTAU_ABS_YEARS_CANDIDATES + PTAU_SIGNED_YEARS_CANDIDATES}, "
            "and PTAU_AGE with AGE."
        )

    return work, {
        "ptau_zscore_source_column": z_col,
        "ptau_interval_source": interval_source,
    }


def validate_binary_column(df: pd.DataFrame, column: str, context: str) -> None:
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    invalid = sorted(set(vals.unique()) - {0, 1})
    if invalid:
        preview = invalid[:10]
        raise ValueError(
            f"{context}: {column} must contain only 0, 1, or missing. "
            f"Invalid values include {preview}."
        )


def prep_dimi(
    df: pd.DataFrame,
    source_name: str,
    region_columns: list[str],
) -> pd.DataFrame:
    required = [SUBJECT_ID_COL, "APOE", "DX", "COPAT", "AGE", *region_columns]
    require_columns(df, required, source_name)

    work = df.copy()
    work["APOE"] = work["APOE"].map(normalize_apoe_value)
    work["DX"] = pd.to_numeric(work["DX"], errors="coerce")
    work["COPAT"] = pd.to_numeric(work["COPAT"], errors="coerce")

    numeric_cols = [
        "AGE", "CL", "zscore_2sd", "days_diff", "PTAU_ZSCORE",
        "PTAU_AGE", "AGE_DIFF_YEARS", "ABS_AGE_DIFF_YEARS",
        *region_columns,
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work[SUBJECT_ID_COL] = work[SUBJECT_ID_COL].astype("string").str.strip()
    work.loc[work[SUBJECT_ID_COL].isin(["", "nan", "None", "<NA>"]), SUBJECT_ID_COL] = pd.NA

    work = work[
        work["APOE"].isin(APOE_ALLOWED)
        & work["DX"].isin(DX_MAP)
        & ~work["COPAT"].eq(1)
        & work[SUBJECT_ID_COL].notna()
    ].copy()

    work["DX_LABEL"] = work["DX"].map(DX_MAP)
    work["E4_DOSE"] = work["APOE"].map(e4_dose)
    work["apoe_key"] = work["APOE"]

    for region_col in region_columns:
        validate_binary_column(work, region_col, source_name)

    return work


def _dedup_eligible(
    df: pd.DataFrame,
    subset: list[str],
    age_col: str = "AGE",
) -> pd.DataFrame:
    keep = "last" if VISIT_SELECTION == "latest" else "first"
    out = (
        df.sort_values([*subset, age_col], kind="mergesort")
        .drop_duplicates(subset=subset, keep=keep)
        .reset_index(drop=True)
    )
    if out.duplicated(subset).any():
        raise AssertionError(f"Deduplication failed for key {subset}")
    return out


def add_title_with_box(ax: plt.Axes, title: str) -> None:
    band_h = 0.20
    ax.add_patch(
        plt.Rectangle(
            xy=(0, 1.0),
            width=1.0,
            height=band_h,
            transform=ax.transAxes,
            facecolor="#EBEBEB",
            edgecolor="black",
            linewidth=plt.rcParams.get("axes.linewidth", 0.8),
            clip_on=False,
            zorder=5,
        )
    )
    ax.text(
        0.5,
        1.0 + band_h * 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=FONT_SIZE["title"],
        clip_on=False,
        zorder=6,
    )


def add_section_label(fig: plt.Figure, y: float, text: str, start_x: float) -> None:
    prefix, remainder = text.split(" ", 1) if " " in text else (text, "")
    x_prefix = max(0.005, start_x + SECTION_LABEL_X_SHIFT)
    x_text = x_prefix + SECTION_LABEL_TEXT_GAP
    fig.text(
        x_prefix,
        y,
        prefix,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE["title"] + 1,
        fontweight="bold",
    )
    if remainder:
        fig.text(
            x_text,
            y,
            remainder,
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE["title"] + 1,
            fontweight="normal",
        )


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, labelsize=FONT_SIZE["tick"])


def grid_y(ax: plt.Axes) -> None:
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)


def positivity_percent(sub: pd.DataFrame, column: str) -> tuple[float, int, int]:
    values = pd.to_numeric(sub[column], errors="coerce").dropna()
    if len(values) == 0:
        return np.nan, 0, 0
    invalid = set(values.unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Non-binary values in {column}: {sorted(invalid)}")
    n = int(len(values))
    k = int(values.sum())
    return float(k / n * 100), n, k


def build_region_samples(
    prepared: pd.DataFrame,
    region_col: str,
    x_type: str,
    audit_rows: list[dict[str, Any]],
    tracer: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    source_name = BASE_FILE.name if x_type == "CL" else PTAU_FILE.name
    initial_n = len(prepared)
    ptau_sources = {
        "ptau_zscore_source_column": "",
        "ptau_interval_source": "",
    }

    if x_type == "CL":
        required = [SUBJECT_ID_COL, "DX_LABEL", "AGE", "CL", region_col, "apoe_key"]
        eligible = prepared.dropna(subset=required).copy()
        eligible["CL_bin_label"] = pd.cut(
            eligible["CL"],
            bins=CL_EDGES,
            labels=CL_BIN_LABELS,
            include_lowest=True,
        ).astype("object")
        eligible = eligible.dropna(subset=["CL_bin_label"])
        max_days: float | int = np.nan
        bin_col = "CL_bin_label"
    elif x_type == "pTau217":
        eligible, ptau_sources = standardize_ptau_columns(prepared, source_name)
        eligible = eligible[
            eligible["ABS_DAYS_DIFF_STD"].le(MAX_DAYS_PTAU_PET)
        ].copy()
        required = [
            SUBJECT_ID_COL,
            "DX_LABEL",
            "AGE",
            "PTAU_ZSCORE_STD",
            "ABS_DAYS_DIFF_STD",
            region_col,
            "apoe_key",
        ]
        eligible = eligible.dropna(subset=required).copy()
        eligible["zscore_bin"] = pd.cut(
            eligible["PTAU_ZSCORE_STD"],
            bins=ZSCORE_EDGES,
            labels=ZSCORE_LABELS,
            include_lowest=True,
        ).astype("object")
        eligible = eligible.dropna(subset=["zscore_bin"])
        max_days = MAX_DAYS_PTAU_PET
        bin_col = "zscore_bin"
    else:
        raise ValueError(f"Unknown x_type: {x_type}")

    by_dx = _dedup_eligible(eligible, [SUBJECT_ID_COL, "DX_LABEL"])
    overall = _dedup_eligible(eligible, [SUBJECT_ID_COL])

    common_meta = {
        **tracer_metadata_columns(tracer),
        "source_file": source_name,
        "section": x_type,
        "region_col": region_col,
        "subject_id_column": SUBJECT_ID_COL,
        "visit_selection": VISIT_SELECTION,
        "max_abs_days_diff": max_days,
        "bin_column": bin_col,
        **ptau_sources,
        "n_after_basic_preparation": int(initial_n),
        "n_eligible_before_dedup": int(len(eligible)),
    }
    audit_rows.append(
        {
            **common_meta,
            "scope": "by_DX",
            "dedup_key": f"{SUBJECT_ID_COL} + DX_LABEL",
            "n_after_dedup": int(len(by_dx)),
            "n_unique_subjects": int(by_dx[SUBJECT_ID_COL].nunique()),
        }
    )
    audit_rows.append(
        {
            **common_meta,
            "scope": "Overall",
            "dedup_key": SUBJECT_ID_COL,
            "n_after_dedup": int(len(overall)),
            "n_unique_subjects": int(overall[SUBJECT_ID_COL].nunique()),
        }
    )

    return {"by_dx": by_dx, "overall": overall}


def draw_line_by_bin(
    ax: plt.Axes,
    data: pd.DataFrame,
    region_col: str,
    bin_col: str,
    bin_labels: list[str],
    x_positions: np.ndarray,
    apoe: str,
) -> None:
    color = palette_apoe[apoe]
    grp = data[data["apoe_key"] == apoe].dropna(subset=[region_col, bin_col])

    xs: list[float | None] = []
    ys: list[float | None] = []
    for xpos, label in zip(x_positions, bin_labels):
        sub = grp[grp[bin_col] == label]
        pct, n, _ = positivity_percent(sub, region_col)
        if n < MIN_N or not np.isfinite(pct):
            xs.append(None)
            ys.append(None)
        else:
            xs.append(float(xpos))
            ys.append(float(pct))

    segment_x: list[float] = []
    segment_y: list[float] = []
    for x_value, y_value in zip(xs, ys):
        if x_value is None:
            if len(segment_x) >= 2:
                ax.plot(
                    segment_x,
                    segment_y,
                    color=color,
                    lw=LW_BIN,
                    zorder=4,
                    alpha=0.95,
                )
            segment_x, segment_y = [], []
        else:
            segment_x.append(x_value)
            segment_y.append(float(y_value))
    if len(segment_x) >= 2:
        ax.plot(
            segment_x,
            segment_y,
            color=color,
            lw=LW_BIN,
            zorder=4,
            alpha=0.95,
        )

    dot_x = [x for x in xs if x is not None]
    dot_y = [y for y in ys if y is not None]
    if dot_x:
        ax.scatter(dot_x, dot_y, color=color, s=MARKER_SIZE, linewidths=0, zorder=5)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    m = len(order)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def global_independence_test(
    table: np.ndarray,
) -> tuple[float, float, float, str]:
    table = np.asarray(table, dtype=int)
    chi2, asymptotic_p, _, expected = chi2_contingency(table, correction=False)
    expected_min = float(np.min(expected))
    if expected_min >= 5:
        return float(chi2), float(asymptotic_p), expected_min, "pearson_chi_square"

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
    return float(chi2), float(p_mc), expected_min, "fixed_margin_monte_carlo_exact"


def corrected_bin_significance(
    data: pd.DataFrame,
    region_col: str,
    bin_col: str,
    bin_labels: list[str],
    x_positions: np.ndarray,
    panel_metadata: dict[str, Any],
) -> tuple[dict[float, bool], list[dict[str, Any]]]:
    bin_records: list[dict[str, Any]] = []

    for xpos, label in zip(x_positions, bin_labels):
        sub_bin = data[data[bin_col] == label].dropna(
            subset=[region_col, "apoe_key"]
        )
        counts: dict[str, dict[str, int]] = {}
        for apoe in apoe_order:
            values = pd.to_numeric(
                sub_bin.loc[sub_bin["apoe_key"] == apoe, region_col],
                errors="coerce",
            ).dropna()
            pos = int(values.sum())
            counts[apoe] = {
                "n": int(len(values)),
                "pos": pos,
                "neg": int(len(values) - pos),
            }

        global_haps = [a for a in apoe_order if counts[a]["n"] >= MIN_N_TEST]
        global_chi2 = np.nan
        global_p_raw = np.nan
        global_expected_min = np.nan
        global_method = "not_tested"
        if len(global_haps) >= 2:
            table = np.asarray(
                [[counts[a]["pos"], counts[a]["neg"]] for a in global_haps],
                dtype=int,
            )
            if table[:, 0].sum() > 0 and table[:, 1].sum() > 0:
                try:
                    (
                        global_chi2,
                        global_p_raw,
                        global_expected_min,
                        global_method,
                    ) = global_independence_test(table)
                except Exception as exc:
                    global_method = f"error:{exc}"

        pair_records: list[dict[str, Any]] = []
        for a1, a2 in combinations(SIGNIFICANCE_APOE_ORDER, 2):
            c1, c2 = counts[a1], counts[a2]
            testable = c1["n"] >= MIN_N_TEST and c2["n"] >= MIN_N_TEST
            odds_ratio = np.nan
            pair_p_raw = np.nan
            note = ""
            if testable:
                table = np.asarray(
                    [[c1["pos"], c1["neg"]], [c2["pos"], c2["neg"]]],
                    dtype=int,
                )
                try:
                    odds_ratio, pair_p_raw = fisher_exact(
                        table, alternative="two-sided"
                    )
                except Exception as exc:
                    note = f"error:{exc}"
            else:
                note = "group_n_below_min"

            pair_records.append(
                {
                    **panel_metadata,
                    "test_level": "pairwise",
                    "bin_label": label,
                    "bin_x": float(xpos),
                    "comparison": f"{a1}_vs_{a2}",
                    "alternative": "two-sided",
                    "n1": c1["n"],
                    "k1": c1["pos"],
                    "n2": c2["n"],
                    "k2": c2["pos"],
                    "odds_ratio": (
                        float(odds_ratio) if np.isfinite(odds_ratio) else np.nan
                    ),
                    "p_raw": (
                        float(pair_p_raw) if np.isfinite(pair_p_raw) else np.nan
                    ),
                    "tested": int(testable and np.isfinite(pair_p_raw)),
                    "note": note,
                    "global_haplotypes": ",".join(global_haps),
                    "global_chi2": global_chi2,
                    "global_p_raw": global_p_raw,
                    "global_expected_min": global_expected_min,
                    "global_test_method": global_method,
                    "global_resamples": (
                        GLOBAL_MONTE_CARLO_RESAMPLES
                        if global_method == "fixed_margin_monte_carlo_exact"
                        else 0
                    ),
                }
            )

        bin_records.append(
            {
                "bin_x": float(xpos),
                "bin_label": label,
                "global_p_raw": global_p_raw,
                "pair_records": pair_records,
            }
        )

    global_adjusted = holm_adjust([b["global_p_raw"] for b in bin_records])
    all_pairs = [r for b in bin_records for r in b["pair_records"]]
    pair_adjusted = holm_adjust([r["p_raw"] for r in all_pairs])

    sig_by_x = {float(x): False for x in x_positions}
    debug_rows: list[dict[str, Any]] = []
    pair_index = 0

    for bin_record, global_p_holm in zip(bin_records, global_adjusted):
        any_pair = False
        for record in bin_record["pair_records"]:
            pair_p_holm = pair_adjusted[pair_index]
            pair_index += 1
            record["global_p_holm_across_bins"] = (
                float(global_p_holm) if np.isfinite(global_p_holm) else np.nan
            )
            record["p_holm_across_bins_and_comparisons"] = (
                float(pair_p_holm) if np.isfinite(pair_p_holm) else np.nan
            )
            record["global_significant_holm"] = bool(
                np.isfinite(global_p_holm) and global_p_holm < ALPHA
            )
            record["pairwise_significant_holm"] = bool(
                np.isfinite(pair_p_holm) and pair_p_holm < ALPHA
            )
            record["significant"] = bool(
                record["global_significant_holm"]
                and record["pairwise_significant_holm"]
            )
            any_pair = any_pair or record["significant"]
            debug_rows.append(record)
        sig_by_x[bin_record["bin_x"]] = bool(any_pair)

    return sig_by_x, debug_rows


def plot_bin_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    region_col: str,
    x_type: str,
    sig_by_x: dict[float, bool],
) -> None:
    if x_type == "CL":
        bin_col = "CL_bin_label"
        bin_labels = CL_BIN_LABELS
        x_positions = CL_BIN_X
        x_axis_label = "Aβ burden (CL)"
    else:
        bin_col = "zscore_bin"
        bin_labels = ZSCORE_LABELS
        x_positions = ZSCORE_BIN_X
        x_axis_label = "Plasma p-tau217 (z)"

    for apoe in apoe_order:
        draw_line_by_bin(
            ax,
            data,
            region_col,
            bin_col,
            bin_labels,
            x_positions,
            apoe,
        )

    for xpos, significant in sig_by_x.items():
        if significant:
            ax.scatter(
                xpos,
                Y_SIG,
                marker="o",
                color="#222222",
                s=24,
                zorder=8,
                clip_on=False,
                linewidths=0,
            )

    ax.set_xlim(0.55, len(bin_labels) + 0.45)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels, fontsize=FONT_SIZE["tick"] - 1.0)
    ax.set_xlabel(x_axis_label, fontsize=FONT_SIZE["axis_label"])
    ax.set_ylim(*YLIM_BIN)
    grid_y(ax)
    style_ax(ax)


print("Loading data...")
if not BASE_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {BASE_FILE}")
if not PTAU_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {PTAU_FILE}")

raw_base = pd.read_csv(BASE_FILE)
raw_ptau = pd.read_csv(PTAU_FILE)
TRACER = resolve_tracer(raw_base, raw_ptau)
OUTPUT_PREFIX = f"FIG_S1_{TRACER['output_tag']}"
REGION_COLUMNS = TRACER["columns"]
BRAAK_REGIONS = list(REGION_COLUMNS.items())

print(f"  Selected tracer: {TRACER['name']}")
print(f"  Regional columns: {list(REGION_COLUMNS.values())}")
print(f"  Output prefix: {OUTPUT_PREFIX}")
print(f"  p-tau/PET window: ±{MAX_DAYS_PTAU_PET} days")

base_prepared = prep_dimi(
    raw_base,
    BASE_FILE.name,
    list(REGION_COLUMNS.values()),
)
ptau_prepared = prep_dimi(
    raw_ptau,
    PTAU_FILE.name,
    list(REGION_COLUMNS.values()),
)
_, PTAU_SOURCE_METADATA = standardize_ptau_columns(ptau_prepared, PTAU_FILE.name)
print(f"  p-tau z-score source: {PTAU_SOURCE_METADATA['ptau_zscore_source_column']}")
print(f"  p-tau interval source: {PTAU_SOURCE_METADATA['ptau_interval_source']}")

sample_selection_audit: list[dict[str, Any]] = []
REGION_SAMPLES: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
for region_label, region_col in BRAAK_REGIONS:
    REGION_SAMPLES[(region_label, "CL")] = build_region_samples(
        base_prepared,
        region_col,
        "CL",
        sample_selection_audit,
        TRACER,
    )
    REGION_SAMPLES[(region_label, "pTau217")] = build_region_samples(
        ptau_prepared,
        region_col,
        "pTau217",
        sample_selection_audit,
        TRACER,
    )


section_labels = {
    "Braak 1–2": rf"a. Braak 1–2 {TRACER['name']} positivity by Aβ burden and plasma p-tau217, clinical status, and $\it{{APOE}}$ haplotype",
    "Braak 3–4": rf"b. Braak 3–4 {TRACER['name']} positivity by Aβ burden and plasma p-tau217, clinical status, and $\it{{APOE}}$ haplotype",
    "Braak 5–6": rf"c. Braak 5–6 {TRACER['name']} positivity by Aβ burden and plasma p-tau217, clinical status, and $\it{{APOE}}$ haplotype",
}

ROW_DEFS = [
    (1, "CL", "Braak 1–2", REGION_COLUMNS["Braak 1–2"]),
    (2, "pTau217", "Braak 1–2", REGION_COLUMNS["Braak 1–2"]),
    (3, "CL", "Braak 3–4", REGION_COLUMNS["Braak 3–4"]),
    (4, "pTau217", "Braak 3–4", REGION_COLUMNS["Braak 3–4"]),
    (5, "CL", "Braak 5–6", REGION_COLUMNS["Braak 5–6"]),
    (6, "pTau217", "Braak 5–6", REGION_COLUMNS["Braak 5–6"]),
]

total_w = 4 * PANEL_W + 3 * COL_GAP
start_x = (1.0 - total_w) / 2
row_bottoms: dict[int, float] = {}
row_bottoms[6] = BOTTOM_MARGIN
row_bottoms[5] = row_bottoms[6] + ROW_H + INTERNAL_ROW_GAP
row_bottoms[4] = row_bottoms[5] + ROW_H + SECTION_GAP
row_bottoms[3] = row_bottoms[4] + ROW_H + INTERNAL_ROW_GAP
row_bottoms[2] = row_bottoms[3] + ROW_H + SECTION_GAP
row_bottoms[1] = row_bottoms[2] + ROW_H + INTERNAL_ROW_GAP

fig = plt.figure(figsize=(20, 15), dpi=300, facecolor="white")
summary_rows: list[dict[str, Any]] = []
region_summary_rows: list[dict[str, Any]] = []
sample_summary_rows: list[dict[str, Any]] = []
significance_debug_rows: list[dict[str, Any]] = []

print("Building figure...")
for row_i, x_type, region_label, region_col in ROW_DEFS:
    samples = REGION_SAMPLES[(region_label, x_type)]
    bin_col = "CL_bin_label" if x_type == "CL" else "zscore_bin"
    bin_labels = CL_BIN_LABELS if x_type == "CL" else ZSCORE_LABELS
    x_positions = CL_BIN_X if x_type == "CL" else ZSCORE_BIN_X
    source_file = BASE_FILE.name if x_type == "CL" else PTAU_FILE.name

    for col_i, dx_label in enumerate(DX_ORDER):
        panel_x = start_x + col_i * (PANEL_W + COL_GAP)
        ax = fig.add_axes([panel_x, row_bottoms[row_i], PANEL_W, ROW_H])
        ax.set_facecolor("white")

        if dx_label == "Overall":
            sub = samples["overall"].copy()
        else:
            sub = samples["by_dx"][
                samples["by_dx"]["DX_LABEL"] == dx_label
            ].copy()

        panel_metadata = {
            **tracer_metadata_columns(TRACER),
            "section": x_type,
            "region": region_label,
            "region_col": region_col,
            "subplot": dx_label,
            "source_file": source_file,
            "subject_id_column": SUBJECT_ID_COL,
            "max_abs_days_diff": (
                MAX_DAYS_PTAU_PET if x_type == "pTau217" else np.nan
            ),
            "ptau_zscore_source_column": (
                PTAU_SOURCE_METADATA["ptau_zscore_source_column"]
                if x_type == "pTau217" else ""
            ),
            "ptau_interval_source": (
                PTAU_SOURCE_METADATA["ptau_interval_source"]
                if x_type == "pTau217" else ""
            ),
        }
        sig_by_x, sig_rows = corrected_bin_significance(
            sub,
            region_col,
            bin_col,
            bin_labels,
            x_positions,
            panel_metadata,
        )
        significance_debug_rows.extend(sig_rows)

        plot_bin_panel(ax, sub, region_col, x_type, sig_by_x)
        add_title_with_box(ax, dx_label)
        ax.set_ylabel(
            "Tau PET positivity (%)",
            fontsize=FONT_SIZE["axis_label"],
            labelpad=8,
        )

        for label, xpos in zip(bin_labels, x_positions):
            sub_bin = sub[sub[bin_col] == label]
            for apoe in [*apoe_order, "ALL"]:
                sub_apoe = (
                    sub_bin
                    if apoe == "ALL"
                    else sub_bin[sub_bin["apoe_key"] == apoe]
                )
                pct, n, k = positivity_percent(sub_apoe, region_col)
                summary_rows.append(
                    {
                        **panel_metadata,
                        "section_label": section_labels[region_label],
                        "x_variable": (
                            "Aβ burden (CL)"
                            if x_type == "CL"
                            else "Plasma p-tau217 z-score"
                        ),
                        "bin_label": label,
                        "bin_x": float(xpos),
                        "APOE": apoe,
                        "n_subjects": n,
                        "n_positive": k,
                        "positivity_pct": pct,
                        "plotted": bool(n >= MIN_N),
                        "low_n_flag": bool(n < MIN_N),
                    }
                )

        for apoe in [*apoe_order, "ALL"]:
            sub_apoe = sub if apoe == "ALL" else sub[sub["apoe_key"] == apoe]
            pct, n, k = positivity_percent(sub_apoe, region_col)
            region_summary_rows.append(
                {
                    **panel_metadata,
                    "APOE": apoe,
                    "n_subjects_with_region": n,
                    "n_positive": k,
                    "overall_positivity_pct": pct,
                    "mean_age": (
                        float(sub_apoe["AGE"].mean()) if len(sub_apoe) else np.nan
                    ),
                    "median_age": (
                        float(sub_apoe["AGE"].median()) if len(sub_apoe) else np.nan
                    ),
                    "mean_CL": (
                        float(sub_apoe["CL"].mean())
                        if "CL" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                    "median_CL": (
                        float(sub_apoe["CL"].median())
                        if "CL" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                    "mean_pTau217_zscore": (
                        float(sub_apoe["PTAU_ZSCORE_STD"].mean())
                        if "PTAU_ZSCORE_STD" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                    "median_pTau217_zscore": (
                        float(sub_apoe["PTAU_ZSCORE_STD"].median())
                        if "PTAU_ZSCORE_STD" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                }
            )

        for apoe in [*apoe_order, "ALL"]:
            sub_apoe = sub if apoe == "ALL" else sub[sub["apoe_key"] == apoe]
            sample_summary_rows.append(
                {
                    **panel_metadata,
                    "APOE": apoe,
                    "n_subjects": int(len(sub_apoe)),
                    "n_unique_subjects": int(sub_apoe[SUBJECT_ID_COL].nunique()),
                    "mean_age": (
                        float(sub_apoe["AGE"].mean()) if len(sub_apoe) else np.nan
                    ),
                    "median_age": (
                        float(sub_apoe["AGE"].median()) if len(sub_apoe) else np.nan
                    ),
                    "mean_CL": (
                        float(sub_apoe["CL"].mean())
                        if "CL" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                    "median_CL": (
                        float(sub_apoe["CL"].median())
                        if "CL" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                    "mean_pTau217_zscore": (
                        float(sub_apoe["PTAU_ZSCORE_STD"].mean())
                        if "PTAU_ZSCORE_STD" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                    "median_pTau217_zscore": (
                        float(sub_apoe["PTAU_ZSCORE_STD"].median())
                        if "PTAU_ZSCORE_STD" in sub_apoe.columns and len(sub_apoe)
                        else np.nan
                    ),
                }
            )


add_section_label(
    fig,
    row_bottoms[1] + ROW_H + SECTION_LABEL_Y_OFFSET,
    section_labels["Braak 1–2"],
    start_x,
)
add_section_label(
    fig,
    row_bottoms[3] + ROW_H + SECTION_LABEL_Y_OFFSET,
    section_labels["Braak 3–4"],
    start_x,
)
add_section_label(
    fig,
    row_bottoms[5] + ROW_H + SECTION_LABEL_Y_OFFSET,
    section_labels["Braak 5–6"],
    start_x,
)

apoe_handles = [
    Line2D([0], [0], color=palette_apoe[a], lw=2.0, label=apoe_labels[a])
    for a in apoe_order
]
fig.legend(
    handles=apoe_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, LEGEND_Y),
    ncol=4,
    frameon=False,
    fontsize=FONT_SIZE["legend"],
    handlelength=2.0,
    columnspacing=1.2,
    handletextpad=0.45,
)


figure_path = Path(f"{OUTPUT_PREFIX}.png")
subplot_path = Path(f"{OUTPUT_PREFIX}_LLM_subplot_summary.csv")
region_path = Path(f"{OUTPUT_PREFIX}_region_summary.csv")
significance_path = Path(f"{OUTPUT_PREFIX}_significance_debug.csv")
sample_path = Path(f"{OUTPUT_PREFIX}_sample_summary.csv")
audit_path = Path(f"{OUTPUT_PREFIX}_sample_selection_audit.csv")

plt.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

pd.DataFrame(summary_rows).to_csv(subplot_path, index=False)
pd.DataFrame(region_summary_rows).to_csv(region_path, index=False)
pd.DataFrame(significance_debug_rows).to_csv(significance_path, index=False)
pd.DataFrame(sample_summary_rows).to_csv(sample_path, index=False)
pd.DataFrame(sample_selection_audit).to_csv(audit_path, index=False)

print(f"Saved: {figure_path}")
print(f"Saved: {subplot_path}")
print(f"Saved: {region_path}")
print(f"Saved: {significance_path}")
print(f"Saved: {sample_path}")
print(f"Saved: {audit_path}")
