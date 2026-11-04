"""
Figure 3. Spatial extent of [18F]flortaucipir tau PET positivity by Aβ burden,
plasma p-tau217, clinical status and APOE haplotype.

Spatial extent is the number of tau-positive Braak regions (Braak I–II, III–IV
and V–VI; range 0–3). The script uses APOE 3/3, 3/4 and 4/4 participants,
excludes co-pathology (COPAT = 1), requires p-tau217 within 365 days of the PET
scan, and keeps the earliest eligible visit per participant (Overall column)
and per participant and diagnosis (CU, MCI and Dementia columns). Centiloid and
tau PET values come from input datasets in which the amyloid and tau PET scans
were already matched in time, so no additional amyloid-to-tau PET interval
filter is applied here. Section a shows stacked bars with the distribution of
spatial extent across Centiloid bins and p-tau217 z-score bins (cells with at
least 5 participants). Section b shows the expected spatial extent from
proportional-odds ordinal logistic regression models (extent ~ biomarker + age
+ sex + APOE ε4 dose), fitted separately for each column, with 95% confidence
bands; the Overall model is also adjusted for diagnosis. Alternative APOE
codings (e4 homozygosity and biomarker × ε4 interaction) are fitted, compared
with likelihood-ratio tests and exported.

Inputs:  DIMI_DATASET_07-09-2026.csv, DIMI_with_ptau217_07-09-2026.csv
Outputs: FIG_3_FTP_first.png and CSV tables with model results, stacked
         proportions, descriptive statistics, model predictions and APOE
         specification tests.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats as scipy_stats
from statsmodels.miscmodels.ordinal_model import OrderedModel


np.random.seed(42)

FONT_SIZE = {
    "panel_label": 20,
    "title": 20,
    "axis_label": 18,
    "tick": 14,
    "legend": 16,
    "n_annot": 9,
}
TITLE_BAND_PTS = 28

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

PALETTE_APOE = {
    "44": "#C7353E",
    "34": "#E0B04A",
    "33": "#389F42",
}

ANALYSIS_APOE = ["33", "34", "44"]
MODEL_APOE = ["33", "34", "44"]
APOE_ORDER = list(ANALYSIS_APOE)
EXCLUDED_APOE = ["22", "23", "24"]
E4_DOSE = {"33": 0.0, "34": 1.0, "44": 2.0}

if set(APOE_ORDER) != set(PALETTE_APOE):
    raise ValueError(
        "APOE_ORDER and PALETTE_APOE disagree. Every plotted genotype needs a "
        "colour, and every colour needs a plotted genotype, or the grouped bars "
        "will reserve slots for genotypes that carry no data."
    )

MODEL_SPECS = {
    "null": {"apoe_terms": [], "interaction": False, "label": "no APOE term"},
    "dose_main": {
        "apoe_terms": ["E4_DOSE"], "interaction": False,
        "label": "epsilon-4 allele dose (0/1/2), main effect",
    },
    "dose_full": {
        "apoe_terms": ["E4_DOSE"], "interaction": True,
        "label": "epsilon-4 allele dose (0/1/2), with predictor interaction",
    },
    "hom_main": {
        "apoe_terms": ["E4_HOM"], "interaction": False,
        "label": "epsilon-4 homozygosity (4/4 vs 3/3+3/4), main effect",
    },
    "hom_full": {
        "apoe_terms": ["E4_HOM"], "interaction": True,
        "label": "epsilon-4 homozygosity (4/4 vs 3/3+3/4), with predictor interaction",
    },
    "_categorical": {
        "apoe_terms": ["APOE_34", "APOE_44"], "interaction": True,
        "label": "DIAGNOSTIC ONLY - unconstrained categorical, never reported",
    },
}
REPORTED_SPECS = ["dose_main", "dose_full", "hom_main", "hom_full"]

SPECIFICATION_FITS: dict[tuple[str, str], dict] = {}
SPECIFICATION_TESTS: list[dict] = []
CURVE_MODEL_SPEC = "dose_main"
APOE_LABELS = {
    "44": r"$\it{APOE}$4/4",
    "34": r"$\it{APOE}$3/4",
    "33": r"$\it{APOE}$3/3",
}

DX_ORDER = ["CU", "MCI", "Dementia"]
DX_MAP = {1: "CU", 2: "MCI", 3: "Dementia"}

OVERALL_LABEL = "Overall"
PANEL_ORDER = [OVERALL_LABEL, *DX_ORDER]

PANEL_SAMPLE_RULE = {
    OVERALL_LABEL: "one eligible row per BIDS_SUBJECT",
    "CU": "one eligible row per BIDS_SUBJECT x DX",
    "MCI": "one eligible row per BIDS_SUBJECT x DX",
    "Dementia": "one eligible row per BIDS_SUBJECT x DX",
}

ADJUST_OVERALL_MODEL_FOR_DX = True

POS_COLS = ["POS_BRAAK12_COH", "POS_BRAAK34_COH", "POS_BRAAK56_COH"]

MAX_EXTENT = len(POS_COLS)
EXTENT_LEVELS = list(range(MAX_EXTENT + 1))

CL_BIN_EDGES = [-np.inf, 25, 50, 75, 100, np.inf]
CL_BIN_LABELS = ["≤25", "25–50", "50–75", "75–100", ">100"]

ZSCORE_BIN_EDGES = [-np.inf, 0, 1.5, 3.0, 4.5, np.inf]
ZSCORE_BIN_LABELS = ["<0", "0–<1.5", "1.5–<3.0", "3.0–<4.5", "≥4.5"]

EXTENT_COLORS = {0: "#ffffff", 1: "#cccccc", 2: "#7f7f7f", 3: "#000000"}
if set(EXTENT_COLORS) != set(EXTENT_LEVELS):
    raise ValueError(
        f"EXTENT_COLORS does not cover the {MAX_EXTENT}-region composite "
        f"(levels {EXTENT_LEVELS})."
    )

ROW_LABELS = {
    1: (
        r"a. Spatial extent of [$^{18}$F]flortaucipir positivity by Aβ PET burden "
        r"and plasma p-tau217 level, clinical status, and $\it{APOE}$"
    ),
    3: (
        r"b. Model-estimated spatial extent of [$^{18}$F]flortaucipir positivity by Aβ PET "
        r"burden and plasma p-tau217 level, clinical status, and $\it{APOE}$"
    ),
}

CL_CURVE_LIMITS = (-10.0, 200.0)
PTAU_CURVE_LIMITS = (-1.0, 8.0)
CL_ANCHORS = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200], dtype=float)
PTAU_ANCHORS = np.array([0, 1.5, 3.0, 4.5, 6.0, 8.0], dtype=float)

MIN_STACKED_N = 5

SHOW_STACKED_N = False

PRINT_STACKED_CELL_COUNTS = True

STACK_HIGHEST_AT_BOTTOM = True

STACKED_SUM_TOLERANCE_PCT = 1e-6

MIN_MODEL_N = 10
MIN_CURVE_GROUP_N = 5
MIN_OUTCOME_LEVELS = 3

FIG_W = 20
FIG_H = 18
ROW_H = {1: 0.16, 2: 0.16, 3: 0.16, 4: 0.16}
BOTTOM_MARGIN = 0.150
GAP_BETWEEN_A_AND_B_SECTION = 0.15
GAP_BETWEEN_A_INTERNAL_ROWS = 0.07
GAP_BETWEEN_B_INTERNAL_ROWS = 0.07

COL_W = 0.200
COL_GAP = 0.06
ROW_LABEL_Y_OFFSET = 0.04
ROW_LABEL_X_SHIFT = -0.03
ROW_LABEL_TEXT_GAP = 0.0225

LEGEND_Y_OFFSET = {1: -0.065, 3: -0.08}
STACKED_LEGEND_BOX_H = 0.02
LINE_LEGEND_BOX_H = 0.05
LEGEND_TITLE_Y = 0.8
LEGEND_CONTENT_Y = 0.05
LEGEND_CONTENT_TEXT_Y = 0.1

BASE_FILE = Path("DIMI_DATASET_07-09-2026.csv")
PTAU_FILE = Path("DIMI_with_ptau217_07-09-2026.csv")
OUTPUT_DIR = Path(".")
FIGURE_NAME = "FIG_3_FTP_first.png"
MAX_DAYS_PTAU_PET = 365.0

AUDIT_CELLS = [
    (1, "CU", ">100"),
    (1, "CU", "75–100"),
    (1, OVERALL_LABEL, ">100"),
]

DATE_SORT_CANDIDATES = [
    "PET_DATE",
    "MK_DATE",
    "TAU_PET_DATE",
    "SCAN_DATE",
    "EXAM_DATE",
    "DATE",
    "SESSION_DATE",
    "VISIT_DATE",
    "AcquisitionDate",
    "acquisition_date",
]


@dataclass
class OrdinalFit:
    panel: str
    predictor_col: str
    predictor_label: str
    predictor_output_name: str
    data_source: str
    status: str
    message: str = ""
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    result: object | None = None
    design_cols: list[str] = field(default_factory=list)
    outcome_levels: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    observed_apoe: list[str] = field(default_factory=list)
    reference_apoe: str | None = None
    spec: str = CURVE_MODEL_SPEC
    age_reference: float = np.nan
    sex_reference: float = np.nan
    fit_warnings: list[str] = field(default_factory=list)
    model_scope: str = "dose_model"
    dx_reference: str | None = None
    dx_weights: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.result is not None

    @property
    def dx_adjusted(self) -> bool:
        return bool(self.dx_weights)

    @property
    def sample_rule(self) -> str:
        return PANEL_SAMPLE_RULE.get(self.panel, "")


@dataclass
class PanelSample:
    overall: pd.DataFrame
    by_dx: pd.DataFrame
    source: str

    def panel(self, label: str) -> pd.DataFrame:
        if label == OVERALL_LABEL:
            return self.overall.copy()
        return self.by_dx.loc[self.by_dx["DX_LABEL"].eq(label)].copy()


def output_path(output_dir: Path, filename: str) -> Path:
    return output_dir / filename


def safe_mean(values: Iterable) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return np.nan if values.empty else float(values.mean())


def safe_median(values: Iterable) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return np.nan if values.empty else float(values.median())


def summary_block(values: Iterable) -> dict[str, float | int]:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def dominant_extent_summary(extent_counts: dict[int, int]) -> tuple[float, float]:
    total = int(sum(extent_counts.values()))
    if total == 0:
        return np.nan, np.nan
    dominant = max(extent_counts, key=extent_counts.get)
    return int(dominant), float(extent_counts[dominant] / total * 100)


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
            raw.loc[text_mask]
            .astype("string")
            .str.lower()
            .str.replace("epsilon", "", regex=False)
            .str.replace("apoe", "", regex=False)
            .str.replace("ε", "", regex=False)
            .str.replace("e", "", regex=False)
            .str.replace(r"[^0-9]", "", regex=True)
        )
        result.loc[text_mask] = cleaned

    result = result.replace({"43": "34", "32": "23", "42": "24"})
    return result


def normalize_sex(values: pd.Series) -> pd.Series:
    raw = values.copy()
    result = pd.Series(pd.NA, index=raw.index, dtype="Float64")

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_present = numeric.dropna()
    if not numeric_present.empty:
        observed = set(float(v) for v in np.unique(numeric_present))
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


def require_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def validate_binary_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    invalid_mask = pd.DataFrame(False, index=df.index, columns=columns)
    for column in columns:
        present = df[column].notna()
        invalid_mask[column] = present & ~df[column].isin([0, 1])

    if invalid_mask.any().any():
        invalid_rows = df.loc[invalid_mask.any(axis=1), columns].head(20)
        raise ValueError(
            f"{dataset_name} contains non-binary regional positivity values. "
            f"Expected only 0, 1, or missing values. First affected rows:\n"
            f"{invalid_rows.to_string(index=False)}"
        )


def choose_dedup_sort(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in DATE_SORT_CANDIDATES:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                return parsed, column

    if "AGE" in df.columns:
        age = pd.to_numeric(df["AGE"], errors="coerce")
        return age, "AGE"

    return pd.Series(np.arange(len(df)), index=df.index), "row_order"


def deduplicate_first_visit(
    df: pd.DataFrame,
    dataset_name: str,
    keys: list[str],
) -> pd.DataFrame:
    before = len(df)
    work = df.copy()
    sort_values, sort_source = choose_dedup_sort(work)
    work["_DEDUP_SORT"] = sort_values
    work["_ORIGINAL_ORDER"] = np.arange(len(work))

    work = (
        work.sort_values(
            [*keys, "_DEDUP_SORT", "_ORIGINAL_ORDER"],
            na_position="last",
        )
        .drop_duplicates(keys, keep="first")
        .drop(columns=["_DEDUP_SORT", "_ORIGINAL_ORDER"])
        .reset_index(drop=True)
    )

    removed = before - len(work)
    print(
        f"  {dataset_name}: removed {removed:,} longitudinal rows within "
        f"{' × '.join(keys)} using {sort_source}."
    )
    return work


def build_panel_sample(
    eligible: pd.DataFrame,
    dataset_name: str,
    bin_column: str,
    bin_source: str,
    bin_edges: list[float],
    bin_labels: list[str],
    right: bool,
) -> PanelSample:
    overall = deduplicate_first_visit(eligible, f"{dataset_name} [Overall]", ["BIDS_SUBJECT"])
    by_dx = deduplicate_first_visit(
        eligible, f"{dataset_name} [by DX]", ["BIDS_SUBJECT", "DX_LABEL"]
    )

    for frame in (overall, by_dx):
        frame[bin_column] = pd.cut(
            frame[bin_source],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=right,
            ordered=True,
        )

    return PanelSample(overall=overall, by_dx=by_dx, source=dataset_name)


def prepare_dataset(df: pd.DataFrame, dataset_name: str, include_ptau: bool) -> pd.DataFrame:
    required = ["BIDS_SUBJECT", "APOE", "COPAT", "DX", "AGE", "SEX", *POS_COLS]
    if include_ptau:
        required.extend(["zscore_2sd", "days_diff"])
    else:
        required.append("CL")
    require_columns(df, required, dataset_name)

    work = df.copy()
    work["APOE"] = normalize_apoe(work["APOE"])
    work["COPAT"] = pd.to_numeric(work["COPAT"], errors="coerce")
    work["DX"] = pd.to_numeric(work["DX"], errors="coerce")
    work["SEX_BINARY"] = normalize_sex(work["SEX"])

    numeric_columns = ["AGE", "CL", "zscore_2sd", "days_diff", "SEX_BINARY", *POS_COLS]
    for column in numeric_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.loc[~work["COPAT"].eq(1)].copy()

    genotyped = work.loc[work["APOE"].notna() & work["DX"].isin(DX_MAP)]
    excluded_mask = genotyped["APOE"].isin(EXCLUDED_APOE)
    if excluded_mask.any():
        counts = genotyped.loc[excluded_mask, "APOE"].value_counts().to_dict()
        print(
            f"  {dataset_name}: excluded {int(excluded_mask.sum()):,} rows carrying an "
            f"epsilon-2 allele ({counts}); none can be placed on a clean epsilon-4 "
            "allele count, so they appear in no panel and in no model."
        )

    unusable = genotyped.loc[
        ~genotyped["APOE"].isin(ANALYSIS_APOE) & ~excluded_mask, "APOE"
    ]
    if not unusable.empty:
        print(
            f"  {dataset_name}: WARNING - {len(unusable):,} rows carry an APOE value "
            f"that is neither an analysed genotype nor a known exclusion "
            f"({unusable.value_counts().head(10).to_dict()}). Check the source coding."
        )

    work = work.loc[work["APOE"].isin(ANALYSIS_APOE) & work["DX"].isin(DX_MAP)].copy()
    work["DX_LABEL"] = work["DX"].map(DX_MAP)
    work["apoe_key"] = work["APOE"].astype("string")
    work["E4_DOSE"] = work["apoe_key"].map(E4_DOSE).astype(float)
    work["E4_HOM"] = work["apoe_key"].eq("44").astype(float)
    work["APOE_34"] = work["apoe_key"].eq("34").astype(float)
    work["APOE_44"] = work["apoe_key"].eq("44").astype(float)

    validate_binary_columns(work, POS_COLS, dataset_name)

    work["spatial"] = work[POS_COLS].sum(axis=1, min_count=len(POS_COLS))

    if include_ptau:
        work["abs_days_diff"] = work["days_diff"].abs()

    return work


def load_analysis_data(
    base_path: Path,
    ptau_path: Path,
    max_days_ptau_pet: float,
) -> tuple[PanelSample, PanelSample]:
    print("Loading and preparing data...")
    print(
        f"  Spatial extent = {' + '.join(POS_COLS)} (0-{MAX_EXTENT}); "
        "MetaROI is not part of the composite."
    )
    print(
        f"  APOE analysed in both the descriptive panels and the models: "
        f"{', '.join(ANALYSIS_APOE)}. Dropped: {', '.join(EXCLUDED_APOE)}."
    )

    base_raw = pd.read_csv(base_path)
    base_eligible = prepare_dataset(base_raw, base_path.name, include_ptau=False)

    base_eligible = base_eligible.dropna(subset=["CL", "spatial", "AGE", "SEX_BINARY"]).copy()
    base = build_panel_sample(
        base_eligible,
        base_path.name,
        bin_column="CL_bin",
        bin_source="CL",
        bin_edges=CL_BIN_EDGES,
        bin_labels=CL_BIN_LABELS,
        right=True,
    )

    ptau_raw = pd.read_csv(ptau_path)
    ptau_eligible = prepare_dataset(ptau_raw, ptau_path.name, include_ptau=True)

    ptau_eligible = ptau_eligible.loc[
        ptau_eligible["abs_days_diff"].le(max_days_ptau_pet)
    ].copy()
    ptau_eligible = ptau_eligible.dropna(subset=["zscore_2sd", "spatial", "AGE", "SEX_BINARY"]).copy()
    ptau = build_panel_sample(
        ptau_eligible,
        ptau_path.name,
        bin_column="zscore_bin",
        bin_source="zscore_2sd",
        bin_edges=ZSCORE_BIN_EDGES,
        bin_labels=ZSCORE_BIN_LABELS,
        right=False,
    )

    print(
        f"  Base CL analysis sample: Overall n = {len(base.overall):,}; "
        f"within-DX rows n = {len(base.by_dx):,}"
    )
    print(
        f"  p-tau217 analysis sample (|days_diff| ≤ {max_days_ptau_pet:g}): "
        f"Overall n = {len(ptau.overall):,}; within-DX rows n = {len(ptau.by_dx):,}"
    )

    for label, sample in [("CL", base), ("p-tau217", ptau)]:
        counts = (
            sample.by_dx.groupby(["DX_LABEL", "apoe_key"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=DX_ORDER, columns=APOE_ORDER, fill_value=0)
        )
        overall_counts = (
            sample.overall.groupby("apoe_key", observed=True)
            .size()
            .reindex(APOE_ORDER, fill_value=0)
            .rename(OVERALL_LABEL)
            .to_frame()
            .T
        )
        counts = pd.concat([overall_counts, counts])
        print(f"\n  {label} sample counts by panel and APOE:")
        print(counts.to_string())
        n_multi_dx = int(
            (sample.by_dx.groupby("BIDS_SUBJECT")["DX_LABEL"].nunique() > 1).sum()
        )
        print(
            f"    {n_multi_dx:,} subjects contribute to more than one diagnosis "
            "column and are counted once in the Overall column."
        )

    return base, ptau


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_SIZE["tick"])


def add_title_with_box(ax: plt.Axes, title: str) -> None:
    fig = ax.get_figure()
    ax_h_pts = ax.get_position().height * fig.get_size_inches()[1] * 72
    band_h = TITLE_BAND_PTS / ax_h_pts
    ax.add_patch(
        Rectangle(
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


def ygrid(ax: plt.Axes, ticks: list[float]) -> None:
    ax.set_yticks(ticks)
    ax.grid(False)


def add_row_label(fig: plt.Figure, y: float, text: str, start_x: float) -> None:
    if " " in text:
        prefix, remainder = text.split(" ", 1)
    else:
        prefix, remainder = text, ""

    x_prefix = max(0.005, start_x + ROW_LABEL_X_SHIFT)
    x_text = x_prefix + ROW_LABEL_TEXT_GAP
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


def add_legend_box_background(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_stacked_legends(
    fig: plt.Figure, y_anchor: float, start_x: float, total_w: float
) -> None:
    box_gap = 0.02 * total_w
    legend_w = (total_w - box_gap) / 2

    extent_ax = fig.add_axes([start_x, y_anchor, legend_w, STACKED_LEGEND_BOX_H])
    add_legend_box_background(extent_ax)
    extent_ax.text(
        0.50,
        LEGEND_TITLE_Y,
        "Spatial extent score",
        ha="center",
        va="center",
        fontsize=FONT_SIZE["legend"],
    )

    box_w = 0.040
    box_h = 0.3
    extent_xs = np.linspace(0.16, 0.80, len(EXTENT_LEVELS))
    for x, extent in zip(extent_xs, EXTENT_LEVELS):
        extent_ax.add_patch(
            Rectangle(
                (x - box_w / 2, LEGEND_CONTENT_Y),
                box_w,
                box_h,
                facecolor=EXTENT_COLORS[extent],
                edgecolor="#9E9E9E",
                linewidth=0.8,
                transform=extent_ax.transAxes,
            )
        )
        extent_ax.text(
            x + box_w / 2 + 0.010,
            LEGEND_CONTENT_TEXT_Y,
            str(extent),
            ha="left",
            va="center",
            fontsize=FONT_SIZE["legend"],
        )

    apoe_ax = fig.add_axes(
        [start_x + legend_w + box_gap, y_anchor, legend_w, STACKED_LEGEND_BOX_H]
    )
    add_legend_box_background(apoe_ax)
    apoe_ax.text(
        0.50,
        LEGEND_TITLE_Y,
        r"$\it{APOE}$ haplotype",
        ha="center",
        va="center",
        fontsize=FONT_SIZE["legend"],
    )

    apoe_xs = np.linspace(0.10, 0.75, len(APOE_ORDER))
    for x, apoe in zip(apoe_xs, APOE_ORDER):
        apoe_ax.add_patch(
            Rectangle(
                (x - box_w / 2, LEGEND_CONTENT_Y),
                box_w,
                box_h,
                facecolor="white",
                edgecolor=PALETTE_APOE[apoe],
                linewidth=1.5,
                transform=apoe_ax.transAxes,
            )
        )
        apoe_ax.text(
            x + box_w / 2 + 0.010,
            LEGEND_CONTENT_TEXT_Y,
            APOE_LABELS[apoe],
            ha="left",
            va="center",
            fontsize=FONT_SIZE["legend"],
        )


def draw_line_legend(
    fig: plt.Figure, y_anchor: float, start_x: float, total_w: float
) -> None:
    center_of_panel_block = start_x + total_w / 2
    legend_box_w = 0.35
    legend_x_start = center_of_panel_block - legend_box_w / 2

    ax = fig.add_axes(
        [legend_x_start, y_anchor, legend_box_w, LINE_LEGEND_BOX_H]
    )
    add_legend_box_background(ax)

    legend_entries = list(MODEL_APOE)
    xs = np.linspace(0.05, 0.72, len(legend_entries))
    for x, apoe in zip(xs, legend_entries):
        line_length = 0.06
        y = LEGEND_CONTENT_TEXT_Y - 0.04
        ax.plot(
            [x, x + line_length],
            [y, y],
            color=PALETTE_APOE[apoe],
            lw=2.2,
            linestyle="-",
        )
        ax.text(
            x + line_length + 0.015,
            y,
            APOE_LABELS[apoe],
            ha="left",
            va="center",
            fontsize=FONT_SIZE["legend"],
        )


def stacked_positions(n_bins: int, n_groups: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    bar_w = 0.2
    small_gap = 0.02
    group_gap = 0.15
    group_w = n_groups * bar_w + (n_groups - 1) * small_gap
    x_centers = np.arange(n_bins) * (group_w + group_gap)
    offsets = (
        np.arange(n_groups, dtype=float) - (n_groups - 1) / 2
    ) * (bar_w + small_gap)
    return bar_w, group_w, x_centers, offsets


def stacked_cell_percentages(subset: pd.DataFrame) -> np.ndarray:
    spatial = pd.to_numeric(subset["spatial"], errors="coerce").dropna()
    if spatial.empty:
        return np.zeros(len(EXTENT_LEVELS), dtype=float)

    percentages = np.asarray(
        [float(spatial.eq(extent).mean() * 100) for extent in EXTENT_LEVELS],
        dtype=float,
    )
    total = percentages.sum()
    if abs(total - 100.0) > STACKED_SUM_TOLERANCE_PCT:
        offending = sorted(set(spatial.unique()) - set(EXTENT_LEVELS))
        raise ValueError(
            f"Stacked cell segments sum to {total:.6f}% instead of 100%. "
            f"Spatial-extent values outside {EXTENT_LEVELS} are present: "
            f"{offending}."
        )
    return percentages


def plot_stacked_by_bin(
    ax: plt.Axes,
    data: pd.DataFrame,
    bin_column: str,
    bin_labels: list[str],
    xlabel: str,
    title: str = "",
    cell_counts: Optional[list[dict]] = None,
) -> None:
    n_bins = len(bin_labels)
    bar_w, group_w, x_centers, offsets = stacked_positions(
        n_bins, len(APOE_ORDER)
    )

    for genotype_index, apoe in enumerate(APOE_ORDER):
        group = data.loc[data["apoe_key"].eq(apoe)].dropna(
            subset=[bin_column, "spatial"]
        )

        bin_subsets = [
            group.loc[group[bin_column].astype("string").eq(label)]
            for label in bin_labels
        ]
        bin_counts = np.asarray([len(subset) for subset in bin_subsets], dtype=int)
        eligible_bins = bin_counts >= MIN_STACKED_N
        percentages = [stacked_cell_percentages(subset) for subset in bin_subsets]
        bottoms = np.zeros(n_bins, dtype=float)

        if cell_counts is not None:
            for label, subset, n_cell, eligible in zip(
                bin_labels, bin_subsets, bin_counts, eligible_bins
            ):
                spatial = pd.to_numeric(subset["spatial"], errors="coerce").dropna()
                record = {
                    "panel": title,
                    "x_bin_label": label,
                    "APOE": apoe,
                    "n": int(n_cell),
                    "displayed": bool(eligible),
                }
                record.update(
                    {
                        f"n_extent_{extent}": int(spatial.eq(extent).sum())
                        for extent in EXTENT_LEVELS
                    }
                )
                cell_counts.append(record)

        draw_order = (
            list(reversed(EXTENT_LEVELS))
            if STACK_HIGHEST_AT_BOTTOM
            else list(EXTENT_LEVELS)
        )
        for extent in draw_order:
            heights = np.asarray(
                [
                    cell[extent] if eligible else 0.0
                    for cell, eligible in zip(percentages, eligible_bins)
                ],
                dtype=float,
            )

            ax.bar(
                x_centers + offsets[genotype_index],
                heights,
                width=bar_w,
                bottom=bottoms,
                color=EXTENT_COLORS[extent],
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
            bottoms += heights

        displayed_totals = bottoms[eligible_bins]
        if displayed_totals.size and not np.allclose(
            displayed_totals, 100.0, atol=1e-6
        ):
            raise ValueError(
                f"Stacked bars for APOE {apoe} in panel '{title}' do not reach "
                f"100%: {np.round(displayed_totals, 6).tolist()}."
            )

        for bin_index, (n_cell, eligible) in enumerate(
            zip(bin_counts, eligible_bins)
        ):
            x = x_centers[bin_index] + offsets[genotype_index]

            if eligible:
                ax.bar(
                    x,
                    bottoms[bin_index],
                    width=bar_w,
                    bottom=0,
                    color="none",
                    edgecolor=PALETTE_APOE[apoe],
                    linewidth=2,
                    zorder=4,
                )

            if SHOW_STACKED_N and n_cell > 0:
                ax.text(
                    x,
                    101.0 if eligible else 2.0,
                    f"n={n_cell}",
                    ha="center",
                    va="bottom" if eligible else "center",
                    rotation=90,
                    fontsize=FONT_SIZE["n_annot"] - 2,
                    color="black" if eligible else "gray",
                    clip_on=False,
                    zorder=5,
                )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(bin_labels, fontsize=FONT_SIZE["tick"] - 3.5)
    ax.set_xlim(
        x_centers[0] - group_w / 2 - 0.1,
        x_centers[-1] + group_w / 2 + 0.1,
    )
    ax.set_ylim(0, 105)
    ygrid(ax, [0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE["axis_label"])
    ax.set_ylabel("Proportion (%)", fontsize=FONT_SIZE["axis_label"], labelpad=4)
    style_ax(ax)
    if title:
        add_title_with_box(ax, title)


def predictor_interaction_name(predictor_col: str, term: str) -> str:
    clean_predictor = "CL" if predictor_col == "CL" else "PTAU"
    return f"{clean_predictor}_x_{term}"


def dx_dummy_name(dx: str) -> str:
    return f"DX_{dx}"


def build_model_design(
    data: pd.DataFrame,
    predictor_col: str,
    spec: str,
    observed_apoe: list[str],
    age_reference: float,
    sex_reference: float,
    dx_reference: str | None = None,
    observed_dx: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    design = pd.DataFrame(index=data.index)
    design[predictor_col] = pd.to_numeric(data[predictor_col], errors="coerce")

    age_centered = pd.to_numeric(data["AGE"], errors="coerce") - age_reference
    if age_centered.nunique(dropna=True) > 1:
        design["AGE_centered"] = age_centered

    sex_centered = pd.to_numeric(data["SEX_BINARY"], errors="coerce") - sex_reference
    if sex_centered.nunique(dropna=True) > 1:
        design["SEX_centered"] = sex_centered

    configuration = MODEL_SPECS[spec]
    for term in configuration["apoe_terms"]:
        values = pd.to_numeric(data[term], errors="coerce").astype(float)
        if values.nunique(dropna=True) < 2:
            continue
        design[term] = values
        if configuration["interaction"]:
            design[predictor_interaction_name(predictor_col, term)] = (
                values * design[predictor_col]
            )

    if dx_reference is not None and observed_dx:
        for dx in observed_dx:
            if dx == dx_reference:
                continue
            design[dx_dummy_name(dx)] = data["DX_LABEL"].eq(dx).astype(float)

    design = design.astype(float)
    return design, list(design.columns)


def fit_ordinal_model(
    data: pd.DataFrame,
    panel: str,
    predictor_col: str,
    predictor_label: str,
    predictor_output_name: str,
    data_source: str,
    adjust_for_dx: bool = False,
    spec: str = CURVE_MODEL_SPEC,
) -> OrdinalFit:
    subset = data.dropna(subset=[predictor_col, "AGE", "SEX_BINARY", "spatial", "apoe_key"]).copy()
    subset["spatial"] = pd.to_numeric(subset["spatial"], errors="coerce")
    subset = subset.loc[subset["spatial"].isin(EXTENT_LEVELS)].copy()
    subset["spatial"] = subset["spatial"].astype(int)

    subset = subset.loc[subset["apoe_key"].isin(MODEL_APOE)].copy()

    observed_apoe = [apoe for apoe in MODEL_APOE if subset["apoe_key"].eq(apoe).any()]
    levels = np.sort(subset["spatial"].unique()).astype(float)

    fit = OrdinalFit(
        panel=panel,
        predictor_col=predictor_col,
        predictor_label=predictor_label,
        predictor_output_name=predictor_output_name,
        data_source=data_source,
        status="failed",
        data=subset,
        outcome_levels=levels,
        observed_apoe=observed_apoe,
    )

    if len(subset) < MIN_MODEL_N:
        fit.message = f"Insufficient sample size: n={len(subset)} < {MIN_MODEL_N}."
        return fit

    if len(levels) < MIN_OUTCOME_LEVELS:
        fit.message = (
            f"Insufficient outcome variation: only {len(levels)} spatial-extent "
            "levels are present."
        )
        return fit

    if len(observed_apoe) < 2:
        fit.message = "Fewer than two APOE groups are present."
        return fit

    reference_apoe = "33" if "33" in observed_apoe else observed_apoe[0]
    age_reference = float(subset["AGE"].median())
    sex_reference = float(pd.to_numeric(subset["SEX_BINARY"], errors="coerce").mean())

    observed_dx: list[str] = []
    dx_reference: str | None = None
    dx_weights: dict[str, float] = {}
    if adjust_for_dx:
        observed_dx = [dx for dx in DX_ORDER if subset["DX_LABEL"].eq(dx).any()]
        if len(observed_dx) >= 2:
            dx_reference = observed_dx[0]
            dx_weights = {
                dx: float(subset["DX_LABEL"].eq(dx).mean()) for dx in observed_dx
            }

    design, design_cols = build_model_design(
        subset,
        predictor_col,
        spec,
        observed_apoe,
        age_reference,
        sex_reference,
        dx_reference=dx_reference,
        observed_dx=observed_dx,
    )

    if not np.isfinite(design.to_numpy()).all():
        fit.message = "The model design matrix contains non-finite values."
        return fit

    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank < design.shape[1]:
        fit.message = (
            f"Rank-deficient design matrix: rank {rank} for {design.shape[1]} columns."
        )
        return fit

    outcome = pd.Categorical(
        subset["spatial"], categories=levels.astype(int).tolist(), ordered=True
    )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = OrderedModel(outcome, design, distr="logit")
            result = model.fit(method="bfgs", maxiter=500, disp=False)
            fit_warnings = [str(item.message) for item in caught]

        converged = bool(result.mle_retvals.get("converged", False))
        if not converged:
            fit.message = "The ordinal model did not converge."
            fit.fit_warnings = fit_warnings
            return fit

        covariance = np.asarray(result.cov_params(), dtype=float)
        if covariance.shape != (len(result.params), len(result.params)):
            fit.message = "Unexpected covariance-matrix dimensions."
            fit.fit_warnings = fit_warnings
            return fit
        if not np.isfinite(covariance).all():
            fit.message = "The fitted covariance matrix contains non-finite values."
            fit.fit_warnings = fit_warnings
            return fit

        fit.status = "ok"
        fit.message = "Model converged."
        fit.result = result
        fit.design_cols = design_cols
        fit.reference_apoe = reference_apoe
        fit.spec = spec
        fit.age_reference = age_reference
        fit.sex_reference = sex_reference
        fit.dx_reference = dx_reference
        fit.dx_weights = dx_weights
        fit.fit_warnings = fit_warnings
        return fit

    except Exception as error:
        fit.message = f"{type(error).__name__}: {error}"
        return fit


def build_prediction_design(
    fit: OrdinalFit,
    predictor_values: np.ndarray,
    apoe: str,
    dx: str | None = None,
) -> pd.DataFrame:
    if not fit.ok:
        raise RuntimeError("Cannot build predictions from a failed model.")
    if apoe not in fit.observed_apoe:
        raise ValueError(f"APOE {apoe} was not observed in the fitted sample.")

    predictor_values = np.asarray(predictor_values, dtype=float)
    design = pd.DataFrame(0.0, index=np.arange(len(predictor_values)), columns=fit.design_cols)
    design[fit.predictor_col] = predictor_values

    if "AGE_centered" in design.columns:
        design["AGE_centered"] = 0.0
    if "SEX_centered" in design.columns:
        design["SEX_centered"] = 0.0

    if apoe not in E4_DOSE:
        raise ValueError(f"APOE {apoe} has no epsilon-4 dose.")
    dose = E4_DOSE[apoe]
    for term, value in (("E4_DOSE", dose), ("E4_HOM", 1.0 if dose >= 2 else 0.0)):
        if term in design.columns:
            design[term] = value
            interaction_name = predictor_interaction_name(fit.predictor_col, term)
            if interaction_name in design.columns:
                design[interaction_name] = value * predictor_values

    if dx is not None and dx != fit.dx_reference:
        column = dx_dummy_name(dx)
        if column in design.columns:
            design[column] = 1.0

    return design.astype(float)


def build_weighted_designs(
    fit: OrdinalFit,
    predictor_values: np.ndarray,
    apoe: str,
) -> list[tuple[float, pd.DataFrame]]:
    if not fit.dx_adjusted:
        return [(1.0, build_prediction_design(fit, predictor_values, apoe))]

    total = float(sum(fit.dx_weights.values()))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Diagnosis weights for the Overall model are unusable.")

    return [
        (
            weight / total,
            build_prediction_design(fit, predictor_values, apoe, dx=dx),
        )
        for dx, weight in fit.dx_weights.items()
    ]


def expected_from_params(
    fit: OrdinalFit,
    params: np.ndarray,
    design: pd.DataFrame,
) -> np.ndarray:
    probabilities = np.asarray(
        fit.result.model.predict(params, exog=design), dtype=float
    )
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    if probabilities.shape[1] != len(fit.outcome_levels):
        raise RuntimeError(
            "Predicted probability columns do not match the observed ordinal levels: "
            f"{probabilities.shape[1]} versus {len(fit.outcome_levels)}."
        )

    return probabilities @ fit.outcome_levels


def expected_from_weighted_designs(
    fit: OrdinalFit,
    params: np.ndarray,
    weighted_designs: list[tuple[float, pd.DataFrame]],
) -> np.ndarray:
    total = None
    for weight, design in weighted_designs:
        contribution = weight * expected_from_params(fit, params, design)
        total = contribution if total is None else total + contribution
    if total is None:
        raise RuntimeError("No prediction designs were supplied.")
    return total


def predict_expected_and_ci(
    fit: OrdinalFit,
    predictor_values: np.ndarray,
    apoe: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weighted_designs = build_weighted_designs(fit, predictor_values, apoe)
    params = np.asarray(fit.result.params, dtype=float)
    covariance = np.asarray(fit.result.cov_params(), dtype=float)

    expected = expected_from_weighted_designs(fit, params, weighted_designs)
    jacobian = np.zeros((len(expected), len(params)), dtype=float)

    for index in range(len(params)):
        step = 1e-5 * max(1.0, abs(params[index]))
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        expected_plus = expected_from_weighted_designs(fit, plus, weighted_designs)
        expected_minus = expected_from_weighted_designs(fit, minus, weighted_designs)
        jacobian[:, index] = (expected_plus - expected_minus) / (2.0 * step)

    variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
    standard_error = np.sqrt(np.clip(variance, 0.0, None))
    lower = expected - 1.96 * standard_error
    upper = expected + 1.96 * standard_error

    return (
        np.clip(expected, 0, MAX_EXTENT),
        np.clip(lower, 0, MAX_EXTENT),
        np.clip(upper, 0, MAX_EXTENT),
    )


def curve_sources(fit: OrdinalFit) -> list[tuple[OrdinalFit, str]]:
    return [(fit, apoe) for apoe in MODEL_APOE if apoe in fit.observed_apoe]


def observed_predictor_range(
    fit: OrdinalFit,
    apoe: str,
    hard_limits: tuple[float, float],
) -> Optional[tuple[float, float]]:
    group = fit.data.loc[fit.data["apoe_key"].eq(apoe), fit.predictor_col]
    group = pd.to_numeric(group, errors="coerce").dropna()
    if len(group) < MIN_CURVE_GROUP_N:
        return None

    lower = max(float(group.min()), hard_limits[0])
    upper = min(float(group.max()), hard_limits[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return lower, upper


def plot_expected_extent(
    ax: plt.Axes,
    fit: OrdinalFit,
    title: str,
    xlim: tuple[float, float],
    xticks: list[float],
) -> None:
    if not fit.ok:
        ax.text(
            0.5,
            0.5,
            f"Model unavailable\n{fit.message}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=10,
            wrap=True,
        )
    else:
        hard_limits = (
            CL_CURVE_LIMITS if fit.predictor_col == "CL" else PTAU_CURVE_LIMITS
        )
        plotted = False
        for source_fit, apoe in curve_sources(fit):
            predictor_range = observed_predictor_range(source_fit, apoe, hard_limits)
            if predictor_range is None:
                continue

            grid = np.linspace(predictor_range[0], predictor_range[1], 200)
            try:
                expected, lower, upper = predict_expected_and_ci(source_fit, grid, apoe)
            except Exception as error:
                print(
                    f"  [prediction] {fit.panel} {fit.predictor_col} APOE {apoe}: {error}"
                )
                continue

            color = PALETTE_APOE[apoe]
            ax.fill_between(grid, lower, upper, color=color, alpha=0.18, zorder=3)
            ax.plot(
                grid,
                expected,
                color=color,
                lw=2,
                linestyle="-",
                zorder=4,
            )
            plotted = True

        if not plotted:
            ax.text(
                0.5,
                0.5,
                f"No APOE group had ≥{MIN_CURVE_GROUP_N} observations\n"
                "with a usable predictor range.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
            )

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.1, MAX_EXTENT + 0.1)
    ygrid(ax, [float(level) for level in EXTENT_LEVELS])
    ax.set_xlabel(fit.predictor_label, fontsize=FONT_SIZE["axis_label"])
    ax.set_ylabel(
        "Expected spatial extent",
        fontsize=FONT_SIZE["axis_label"],
        labelpad=10,
    )
    style_ax(ax)
    add_title_with_box(ax, title)


def likelihood_ratio_test(full: OrdinalFit, reduced: OrdinalFit) -> Optional[dict]:
    if not (full.ok and reduced.ok):
        return None
    df = len(full.design_cols) - len(reduced.design_cols)
    if df <= 0:
        return None
    try:
        statistic = 2.0 * (float(full.result.llf) - float(reduced.result.llf))
    except Exception:
        return None
    if not np.isfinite(statistic):
        return None
    statistic = max(statistic, 0.0)
    return {
        "statistic": statistic,
        "df": int(df),
        "p_value": float(scipy_stats.chi2.sf(statistic, df)),
    }


def fit_specification_set(
    data: pd.DataFrame,
    panel: str,
    predictor_col: str,
    predictor_label: str,
    predictor_output_name: str,
    data_source: str,
    adjust_for_dx: bool,
) -> tuple[dict[str, OrdinalFit], list[dict]]:
    fits = {
        spec: fit_ordinal_model(
            data,
            panel=panel,
            predictor_col=predictor_col,
            predictor_label=predictor_label,
            predictor_output_name=predictor_output_name,
            data_source=data_source,
            adjust_for_dx=adjust_for_dx,
            spec=spec,
        )
        for spec in MODEL_SPECS
    }

    tests: list[dict] = []

    def record(full_spec, reduced_spec, name, description):
        test = likelihood_ratio_test(fits[full_spec], fits[reduced_spec])
        if test:
            tests.append({
                "panel": panel,
                "model_predictor": predictor_output_name,
                "test": name,
                "description": description,
                "chi_square": test["statistic"],
                "df": test["df"],
                "p_value": test["p_value"],
                "significant_at_0.05": bool(test["p_value"] < 0.05),
                "data_source": data_source,
            })

    record("dose_main", "null", "e4_dose_effect_LRT",
           "Does epsilon-4 allele count add to the model?")
    record("hom_main", "null", "e4_homozygosity_effect_LRT",
           "Does epsilon-4 homozygosity add to the model?")
    record("dose_full", "dose_main", "e4_dose_x_predictor_LRT",
           "Does the epsilon-4 dose effect vary with the biomarker?")
    record("hom_full", "hom_main", "e4_homozygosity_x_predictor_LRT",
           "Does the homozygosity effect vary with the biomarker?")
    record("_categorical", "dose_full", "dose_linearity_LRT",
           "Significant means the effect is NOT linear in epsilon-4 allele count "
           "in this panel; do not write 'per allele' here.")
    record("_categorical", "hom_full", "homozygosity_adequacy_LRT",
           "Significant means pooling APOE3/3 with APOE3/4 discards real signal "
           "in this panel.")
    return fits, tests


def fit_all_models(
    base: PanelSample,
    ptau: PanelSample,
) -> tuple[dict[str, OrdinalFit], dict[str, OrdinalFit]]:
    cl_models: dict[str, OrdinalFit] = {}
    ptau_models: dict[str, OrdinalFit] = {}

    print("\nFitting ordinal models...")
    if ADJUST_OVERALL_MODEL_FOR_DX:
        print(
            "  Overall models are stage-adjusted and marginalized over the "
            "observed diagnosis composition."
        )
    else:
        print("  Overall models are marginal with respect to clinical stage.")

    for panel in PANEL_ORDER:
        adjust = panel == OVERALL_LABEL and ADJUST_OVERALL_MODEL_FOR_DX

        cl_fits, cl_tests = fit_specification_set(
            base.panel(panel),
            panel=panel,
            predictor_col="CL",
            predictor_label="Aβ burden (CL)",
            predictor_output_name="Abeta_burden_CL",
            data_source=base.source,
            adjust_for_dx=adjust,
        )
        SPECIFICATION_FITS[("Abeta_burden_CL", panel)] = cl_fits
        SPECIFICATION_TESTS.extend(cl_tests)
        cl_fit = cl_fits[CURVE_MODEL_SPEC]
        cl_models[panel] = cl_fit
        print(
            f"  CL | {panel}: {cl_fit.status} — {cl_fit.message} "
            f"(n={len(cl_fit.data)})"
        )
        for warning_message in cl_fit.fit_warnings:
            print(f"    warning: {warning_message}")

        ptau_fits, ptau_tests = fit_specification_set(
            ptau.panel(panel),
            panel=panel,
            predictor_col="zscore_2sd",
            predictor_label="Plasma p-tau217 (z)",
            predictor_output_name="pTau217_zscore",
            data_source=ptau.source,
            adjust_for_dx=adjust,
        )
        SPECIFICATION_FITS[("pTau217_zscore", panel)] = ptau_fits
        SPECIFICATION_TESTS.extend(ptau_tests)
        ptau_fit = ptau_fits[CURVE_MODEL_SPEC]
        ptau_models[panel] = ptau_fit
        print(
            f"  p-tau217 | {panel}: {ptau_fit.status} — {ptau_fit.message} "
            f"(n={len(ptau_fit.data)})"
        )
        for warning_message in ptau_fit.fit_warnings:
            print(f"    warning: {warning_message}")

    return cl_models, ptau_models


def build_figure(
    base: PanelSample,
    ptau: PanelSample,
    cl_models: dict[str, OrdinalFit],
    ptau_models: dict[str, OrdinalFit],
    figure_path: Path,
) -> None:
    print("\nBuilding figure...")

    n_columns = len(PANEL_ORDER)
    total_w = n_columns * COL_W + (n_columns - 1) * COL_GAP
    if total_w > 1.0:
        raise ValueError(
            f"The panel block is {total_w:.3f} of the figure width. Reduce COL_W "
            "or COL_GAP."
        )
    start_x = (1.0 - total_w) / 2

    row_bottoms: dict[int, float] = {}
    row_bottoms[4] = BOTTOM_MARGIN
    row_bottoms[3] = row_bottoms[4] + ROW_H[4] + GAP_BETWEEN_B_INTERNAL_ROWS
    row_bottoms[2] = row_bottoms[3] + ROW_H[3] + GAP_BETWEEN_A_AND_B_SECTION
    row_bottoms[1] = row_bottoms[2] + ROW_H[2] + GAP_BETWEEN_A_INTERNAL_ROWS

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300, facecolor="white")
    cell_counts: list[dict] = []

    for row in [1, 2, 3, 4]:
        for column_index, panel in enumerate(PANEL_ORDER):
            panel_x = start_x + column_index * (COL_W + COL_GAP)
            ax = fig.add_axes([panel_x, row_bottoms[row], COL_W, ROW_H[row]])
            ax.set_facecolor("white")

            if row == 1:
                plot_stacked_by_bin(
                    ax,
                    base.panel(panel),
                    bin_column="CL_bin",
                    bin_labels=CL_BIN_LABELS,
                    xlabel="Aβ burden (CL)",
                    title=panel,
                    cell_counts=cell_counts,
                )
            elif row == 2:
                plot_stacked_by_bin(
                    ax,
                    ptau.panel(panel),
                    bin_column="zscore_bin",
                    bin_labels=ZSCORE_BIN_LABELS,
                    xlabel="Plasma p-tau217 (z)",
                    title=panel,
                    cell_counts=cell_counts,
                )
            elif row == 3:
                plot_expected_extent(
                    ax,
                    cl_models[panel],
                    title=panel,
                    xlim=(-15, 215),
                    xticks=[0, 50, 100, 150, 200],
                )
            else:
                plot_expected_extent(
                    ax,
                    ptau_models[panel],
                    title=panel,
                    xlim=(-1, 8),
                    xticks=[0, 2, 4, 6, 8],
                )

    add_row_label(
        fig,
        row_bottoms[1] + ROW_H[1] + ROW_LABEL_Y_OFFSET,
        ROW_LABELS[1],
        start_x,
    )
    add_row_label(
        fig,
        row_bottoms[3] + ROW_H[3] + ROW_LABEL_Y_OFFSET,
        ROW_LABELS[3],
        start_x,
    )

    fig.canvas.draw()
    draw_stacked_legends(
        fig,
        y_anchor=row_bottoms[2] + LEGEND_Y_OFFSET[1],
        start_x=start_x,
        total_w=total_w,
    )
    draw_line_legend(
        fig,
        y_anchor=row_bottoms[4] + LEGEND_Y_OFFSET[3],
        start_x=start_x,
        total_w=total_w,
    )

    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {figure_path}")

    if PRINT_STACKED_CELL_COUNTS and cell_counts:
        table = pd.DataFrame(cell_counts)
        suppressed = int((~table["displayed"]).sum())
        print(
            f"\n  Stacked-cell support (MIN_STACKED_N = {MIN_STACKED_N}); "
            f"{suppressed:,} of {len(table):,} cells fall below it and are left "
            "blank in the figure:"
        )
        print(table.to_string(index=False))


def append_llm_stacked_row(
    records: list[dict],
    figure_row: int,
    row_label: str,
    panel: str,
    x_variable: str,
    bin_label: str,
    apoe: str,
    subset: pd.DataFrame,
    data_source: str,
    extra: Optional[dict] = None,
) -> None:
    extra = extra or {}
    spatial = pd.to_numeric(subset["spatial"], errors="coerce").dropna()
    counts = {extent: int(spatial.eq(extent).sum()) for extent in EXTENT_LEVELS}
    n_total = int(len(spatial))
    dominant_extent, dominant_pct = dominant_extent_summary(counts)

    record = {
        "figure_row": figure_row,
        "row_label": row_label,
        "subplot": panel,
        "panel": panel,
        "panel_sample": PANEL_SAMPLE_RULE.get(panel, ""),
        "panel_type": "stacked_proportion",
        "model_predictor": np.nan,
        "x_variable": x_variable,
        "x_bin_label": bin_label,
        "APOE": apoe,
        "n_subjects": n_total,
        "min_stacked_n": MIN_STACKED_N,
        "displayed_in_stacked_figure": bool(n_total >= MIN_STACKED_N),
        "low_n_flag": bool(n_total < MIN_STACKED_N),
        "very_low_n_flag": bool(n_total < 2),
        "mean_spatial_extent": safe_mean(spatial),
        "median_spatial_extent": safe_median(spatial),
        **{f"n_extent_{extent}": counts[extent] for extent in EXTENT_LEVELS},
        **{
            f"pct_extent_{extent}": (
                np.nan if n_total == 0 else float(counts[extent] / n_total * 100)
            )
            for extent in EXTENT_LEVELS
        },
        "pct_extent_ge_1": np.nan if n_total == 0 else float(spatial.ge(1).mean() * 100),
        "pct_extent_ge_2": np.nan if n_total == 0 else float(spatial.ge(2).mean() * 100),
        "pct_extent_ge_3": np.nan if n_total == 0 else float(spatial.ge(3).mean() * 100),
        "max_extent": MAX_EXTENT,
        "pct_extent_eq_max": (
            np.nan if n_total == 0 else float(spatial.eq(MAX_EXTENT).mean() * 100)
        ),
        "dominant_extent": dominant_extent,
        "dominant_extent_pct": dominant_pct,
        "expected_extent_ci_lo": np.nan,
        "expected_extent_ci_hi": np.nan,
        "data_source": data_source,
    }
    record.update(extra)
    records.append(record)


def generate_stacked_exports(
    base: PanelSample,
    ptau: PanelSample,
    max_days_ptau_pet: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    proportion_rows: list[dict] = []
    llm_rows: list[dict] = []
    row_summary_rows: list[dict] = []

    configurations = [
        {
            "sample": base,
            "figure_row": 1,
            "bin_col": "CL_bin",
            "bin_labels": CL_BIN_LABELS,
            "x_variable": "Abeta_burden_CL_bin",
            "predictor_col": "CL",
            "predictor_output": "Abeta_burden_CL",
            "source": base.source,
            "extra": {},
        },
        {
            "sample": ptau,
            "figure_row": 2,
            "bin_col": "zscore_bin",
            "bin_labels": ZSCORE_BIN_LABELS,
            "x_variable": "pTau217_zscore_bin",
            "predictor_col": "zscore_2sd",
            "predictor_output": "pTau217_zscore",
            "source": ptau.source,
            "extra": {"max_abs_days_diff_included": float(max_days_ptau_pet)},
        },
    ]

    for config in configurations:
        sample: PanelSample = config["sample"]
        for panel in PANEL_ORDER:
            panel_data = sample.panel(panel)
            for apoe in APOE_ORDER + ["ALL"]:
                apoe_data = (
                    panel_data
                    if apoe == "ALL"
                    else panel_data.loc[panel_data["apoe_key"].eq(apoe)]
                )
                spatial = pd.to_numeric(apoe_data["spatial"], errors="coerce").dropna()

                summary = {
                    "figure_row": config["figure_row"],
                    "row_label": ROW_LABELS[1],
                    "subplot": panel,
                    "panel": panel,
                    "panel_sample": PANEL_SAMPLE_RULE.get(panel, ""),
                    "APOE": apoe,
                    "model_predictor": config["predictor_output"],
                    "n_subjects": int(len(spatial)),
                    "mean_spatial_extent": safe_mean(spatial),
                    "median_spatial_extent": safe_median(spatial),
                    "pct_extent_ge_1": np.nan if spatial.empty else float(spatial.ge(1).mean() * 100),
                    "pct_extent_ge_2": np.nan if spatial.empty else float(spatial.ge(2).mean() * 100),
                    "max_extent": MAX_EXTENT,
                    "pct_extent_eq_max": (
                        np.nan
                        if spatial.empty
                        else float(spatial.eq(MAX_EXTENT).mean() * 100)
                    ),
                    f"mean_{config['predictor_output']}": safe_mean(apoe_data[config["predictor_col"]]),
                    f"median_{config['predictor_output']}": safe_median(apoe_data[config["predictor_col"]]),
                    "data_source": config["source"],
                    **config["extra"],
                }
                row_summary_rows.append(summary)

                valid_binned = apoe_data.dropna(subset=[config["bin_col"], "spatial"])
                for bin_label in config["bin_labels"]:
                    subset = valid_binned.loc[
                        valid_binned[config["bin_col"]].astype("string").eq(bin_label)
                    ]
                    spatial_bin = pd.to_numeric(subset["spatial"], errors="coerce").dropna()
                    n_total = int(len(spatial_bin))
                    counts = {
                        extent: int(spatial_bin.eq(extent).sum())
                        for extent in EXTENT_LEVELS
                    }

                    if apoe != "ALL":
                        for extent in EXTENT_LEVELS:
                            proportion_rows.append(
                                {
                                    "figure_row": config["figure_row"],
                                    "row_label": ROW_LABELS[1],
                                    "DX": panel,
                                    "panel": panel,
                                    "panel_sample": PANEL_SAMPLE_RULE.get(panel, ""),
                                    "APOE": apoe,
                                    "model_predictor": config["predictor_output"],
                                    "x_variable": config["x_variable"],
                                    "x_bin_label": bin_label,
                                    "extent": extent,
                                    "n_total": n_total,
                                    "n_extent": counts[extent],
                                    "pct_extent": (
                                        np.nan
                                        if n_total == 0
                                        else float(counts[extent] / n_total * 100)
                                    ),
                                    "min_stacked_n": MIN_STACKED_N,
                                    "displayed_in_stacked_figure": bool(
                                        n_total >= MIN_STACKED_N
                                    ),
                                    "low_n_flag": bool(n_total < MIN_STACKED_N),
                                    "very_low_n_flag": bool(n_total < 2),
                                    "data_source": config["source"],
                                    **config["extra"],
                                }
                            )

                    extra = {
                        f"mean_{config['predictor_output']}_in_bin": safe_mean(
                            subset[config["predictor_col"]]
                        ),
                        f"median_{config['predictor_output']}_in_bin": safe_median(
                            subset[config["predictor_col"]]
                        ),
                        **config["extra"],
                    }
                    append_llm_stacked_row(
                        llm_rows,
                        figure_row=config["figure_row"],
                        row_label=ROW_LABELS[1],
                        panel=panel,
                        x_variable=config["x_variable"],
                        bin_label=bin_label,
                        apoe=apoe,
                        subset=subset,
                        data_source=config["source"],
                        extra=extra,
                    )

    return proportion_rows, llm_rows, row_summary_rows


def curve_features(
    x_values: np.ndarray,
    expected_values: np.ndarray,
    predictor_name: str,
) -> dict[str, float | str]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(expected_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    result: dict[str, float | str] = {
        "model_predictor": predictor_name,
        "predictor_min_curve": np.nan,
        "predictor_max_curve": np.nan,
        "expected_extent_at_min_predictor": np.nan,
        "expected_extent_at_max_predictor": np.nan,
        "delta_expected_extent_min_to_max_predictor": np.nan,
        "mean_expected_extent_curve": np.nan,
        "max_expected_extent_curve": np.nan,
        "min_expected_extent_curve": np.nan,
        "approx_slope_low_predictor": np.nan,
        "approx_slope_high_predictor": np.nan,
    }
    if x.size == 0:
        return result

    result.update(
        {
            "predictor_min_curve": float(x.min()),
            "predictor_max_curve": float(x.max()),
            "expected_extent_at_min_predictor": float(y[0]),
            "expected_extent_at_max_predictor": float(y[-1]),
            "delta_expected_extent_min_to_max_predictor": float(y[-1] - y[0]),
            "mean_expected_extent_curve": float(y.mean()),
            "max_expected_extent_curve": float(y.max()),
            "min_expected_extent_curve": float(y.min()),
        }
    )

    if len(x) >= 5:
        low_index = min(4, len(x) - 1)
        high_index = max(0, len(x) - 5)
        if x[low_index] != x[0]:
            result["approx_slope_low_predictor"] = float(
                (y[low_index] - y[0]) / (x[low_index] - x[0])
            )
        if x[-1] != x[high_index]:
            result["approx_slope_high_predictor"] = float(
                (y[-1] - y[high_index]) / (x[-1] - x[high_index])
            )

    if predictor_name == "Abeta_burden_CL":
        result.update(
            {
                "cl_min_curve": result["predictor_min_curve"],
                "cl_max_curve": result["predictor_max_curve"],
                "expected_extent_at_min_cl": result[
                    "expected_extent_at_min_predictor"
                ],
                "expected_extent_at_max_cl": result[
                    "expected_extent_at_max_predictor"
                ],
                "delta_expected_extent_min_to_max": result[
                    "delta_expected_extent_min_to_max_predictor"
                ],
                "approx_slope_low_cl": result["approx_slope_low_predictor"],
                "approx_slope_high_cl": result["approx_slope_high_predictor"],
            }
        )

    return result


def model_result_rows(
    fits: list[OrdinalFit],
) -> tuple[list[dict], list[dict]]:
    wide_rows: list[dict] = []
    long_rows: list[dict] = []

    for fit in fits:
        wide = {
            "DX": fit.panel,
            "panel": fit.panel,
            "panel_sample": fit.sample_rule,
            "model_scope": fit.model_scope,
            "model_predictor": fit.predictor_output_name,
            "data_source": fit.data_source,
            "model_status": fit.status,
            "model_message": fit.message,
            "n_subjects": int(len(fit.data)),
            "n_outcome_levels": int(len(fit.outcome_levels)),
            "outcome_levels": ",".join(str(int(x)) for x in fit.outcome_levels),
            "observed_APOE": ",".join(fit.observed_apoe),
            "reference_APOE": fit.reference_apoe,
            "age_reference": fit.age_reference,
            "sex_reference_male_proportion": fit.sex_reference,
            "dx_adjusted": bool(fit.dx_adjusted),
            "reference_DX": fit.dx_reference,
            "dx_marginalization_weights": (
                "; ".join(f"{dx}={weight:.6f}" for dx, weight in fit.dx_weights.items())
                if fit.dx_weights
                else ""
            ),
            "design_columns": ",".join(fit.design_cols),
            "fit_warnings": " | ".join(fit.fit_warnings),
        }

        if fit.ok:
            for term, coefficient, p_value in zip(
                fit.result.params.index,
                fit.result.params.values,
                fit.result.pvalues.values,
            ):
                wide[f"coef_{term}"] = float(coefficient)
                wide[f"pval_{term}"] = float(p_value)
                long_rows.append(
                    {
                        "figure_row": 3 if fit.predictor_col == "CL" else 4,
                        "row_label": ROW_LABELS[3],
                        "subplot": fit.panel,
                        "panel": fit.panel,
                        "panel_sample": fit.sample_rule,
                        "model_scope": fit.model_scope,
                        "model_family": "proportional_odds_ordinal_logit",
                        "model_predictor": fit.predictor_output_name,
                        "outcome": f"spatial_extent_0_to_{MAX_EXTENT}",
                        "term": term,
                        "coefficient": float(coefficient),
                        "p_value": float(p_value),
                        "n_subjects": int(len(fit.data)),
                        "reference_APOE": fit.reference_apoe,
                        "age_reference": fit.age_reference,
                        "sex_reference_male_proportion": fit.sex_reference,
                        "data_source": fit.data_source,
                    }
                )
        wide_rows.append(wide)

    return wide_rows, long_rows


def generate_model_prediction_exports(
    fits: list[OrdinalFit],
    max_days_ptau_pet: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    anchor_rows: list[dict] = []
    full_grid_rows: list[dict] = []
    llm_rows: list[dict] = []
    curve_summary_rows: list[dict] = []
    row_summary_rows: list[dict] = []

    for fit in fits:
        figure_row = 3 if fit.predictor_col == "CL" else 4
        hard_limits = CL_CURVE_LIMITS if fit.predictor_col == "CL" else PTAU_CURVE_LIMITS
        anchors = CL_ANCHORS if fit.predictor_col == "CL" else PTAU_ANCHORS

        if not fit.ok:
            continue

        for source_fit, apoe in curve_sources(fit):
            model_scope = source_fit.model_scope
            group = source_fit.data.loc[source_fit.data["apoe_key"].eq(apoe)].copy()
            predictor_range = observed_predictor_range(source_fit, apoe, hard_limits)
            if predictor_range is None:
                continue

            grid = np.linspace(predictor_range[0], predictor_range[1], 200)
            try:
                expected_grid, lower_grid, upper_grid = predict_expected_and_ci(
                    source_fit, grid, apoe
                )
                expected_anchor, lower_anchor, upper_anchor = predict_expected_and_ci(
                    source_fit, anchors, apoe
                )
            except Exception as error:
                print(
                    f"  [export prediction] {fit.panel} {fit.predictor_col} APOE {apoe}: "
                    f"{error}"
                )
                continue

            extra = (
                {"max_abs_days_diff_included": float(max_days_ptau_pet)}
                if fit.predictor_col != "CL"
                else {}
            )

            for anchor, expected, lower, upper in zip(
                anchors, expected_anchor, lower_anchor, upper_anchor
            ):
                in_range = predictor_range[0] <= anchor <= predictor_range[1]
                row = {
                    "figure_row": figure_row,
                    "row_label": ROW_LABELS[3],
                    "DX": fit.panel,
                    "panel": fit.panel,
                    "panel_sample": fit.sample_rule,
                    "APOE": apoe,
                    "model_predictor": fit.predictor_output_name,
                    "Abeta_burden_CL": float(anchor) if fit.predictor_col == "CL" else np.nan,
                    "pTau217_zscore": float(anchor) if fit.predictor_col != "CL" else np.nan,
                    "expected_extent": float(expected) if in_range else np.nan,
                    "expected_extent_ci_lo": float(lower) if in_range else np.nan,
                    "expected_extent_ci_hi": float(upper) if in_range else np.nan,
                    "age_used_median": source_fit.age_reference,
                    "sex_used_mean_male": source_fit.sex_reference,
                    "n_subjects": int(len(group)),
                    "in_data_range": int(in_range),
                    "reference_APOE": source_fit.reference_apoe,
                    "model_scope": model_scope,
                    "data_source": fit.data_source,
                    **extra,
                }
                anchor_rows.append(row)

                llm_rows.append(
                    {
                        "figure_row": figure_row,
                        "row_label": ROW_LABELS[3],
                        "subplot": fit.panel,
                        "panel": fit.panel,
                        "panel_sample": fit.sample_rule,
                        "panel_type": "ordinal_model_expected_extent",
                        "model_predictor": fit.predictor_output_name,
                        "x_variable": fit.predictor_output_name,
                        "x_bin_label": f"{anchor:g}",
                        "APOE": apoe,
                        "n_subjects": int(len(group)),
                        "low_n_flag": bool(len(group) < MIN_CURVE_GROUP_N),
                        "very_low_n_flag": bool(len(group) < 3),
                        "mean_spatial_extent": float(expected) if in_range else np.nan,
                        "median_spatial_extent": np.nan,
                        **{f"n_extent_{extent}": np.nan for extent in EXTENT_LEVELS},
                        **{f"pct_extent_{extent}": np.nan for extent in EXTENT_LEVELS},
                        "pct_extent_ge_1": np.nan,
                        "pct_extent_ge_2": np.nan,
                        "pct_extent_ge_3": np.nan,
                        "max_extent": MAX_EXTENT,
                        "pct_extent_eq_max": np.nan,
                        "dominant_extent": np.nan,
                        "dominant_extent_pct": np.nan,
                        "expected_extent_ci_lo": float(lower) if in_range else np.nan,
                        "expected_extent_ci_hi": float(upper) if in_range else np.nan,
                        "age_used_median": source_fit.age_reference,
                        "sex_used_mean_male": source_fit.sex_reference,
                        "reference_APOE": source_fit.reference_apoe,
                        "model_scope": model_scope,
                        "data_source": fit.data_source,
                        **extra,
                    }
                )

            for x_value, expected, lower, upper in zip(
                grid, expected_grid, lower_grid, upper_grid
            ):
                full_grid_rows.append(
                    {
                        "figure_row": figure_row,
                        "row_label": ROW_LABELS[3],
                        "DX": fit.panel,
                        "panel": fit.panel,
                        "panel_sample": fit.sample_rule,
                        "APOE": apoe,
                        "model_predictor": fit.predictor_output_name,
                        "Abeta_burden_CL": float(x_value) if fit.predictor_col == "CL" else np.nan,
                        "pTau217_zscore": float(x_value) if fit.predictor_col != "CL" else np.nan,
                        "expected_extent": float(expected),
                        "expected_extent_ci_lo": float(lower),
                        "expected_extent_ci_hi": float(upper),
                        "age_used_median": source_fit.age_reference,
                        "sex_used_mean_male": source_fit.sex_reference,
                        "n_subjects": int(len(group)),
                        "reference_APOE": source_fit.reference_apoe,
                        "model_scope": model_scope,
                        "data_source": fit.data_source,
                        **extra,
                    }
                )

            features = curve_features(grid, expected_grid, fit.predictor_output_name)
            curve_summary_rows.append(
                {
                    "figure_row": figure_row,
                    "row_label": ROW_LABELS[3],
                    "subplot": fit.panel,
                    "panel": fit.panel,
                    "panel_sample": fit.sample_rule,
                    "APOE": apoe,
                    "n_subjects": int(len(group)),
                    "age_used_median": source_fit.age_reference,
                    "sex_used_mean_male": source_fit.sex_reference,
                    "reference_APOE": source_fit.reference_apoe,
                    "model_scope": model_scope,
                    f"mean_observed_{fit.predictor_output_name}": safe_mean(
                        group[fit.predictor_col]
                    ),
                    f"median_observed_{fit.predictor_output_name}": safe_median(
                        group[fit.predictor_col]
                    ),
                    "mean_observed_spatial_extent": safe_mean(group["spatial"]),
                    "median_observed_spatial_extent": safe_median(group["spatial"]),
                    **features,
                    "data_source": fit.data_source,
                    **extra,
                }
            )

            spatial = pd.to_numeric(group["spatial"], errors="coerce").dropna()
            row_summary_rows.append(
                {
                    "figure_row": figure_row,
                    "row_label": ROW_LABELS[3],
                    "subplot": fit.panel,
                    "panel": fit.panel,
                    "panel_sample": fit.sample_rule,
                    "APOE": apoe,
                    "model_predictor": fit.predictor_output_name,
                    "n_subjects": int(len(spatial)),
                    "mean_spatial_extent": safe_mean(spatial),
                    "median_spatial_extent": safe_median(spatial),
                    "pct_extent_ge_1": np.nan if spatial.empty else float(spatial.ge(1).mean() * 100),
                    "pct_extent_ge_2": np.nan if spatial.empty else float(spatial.ge(2).mean() * 100),
                    "max_extent": MAX_EXTENT,
                    "pct_extent_eq_max": (
                        np.nan
                        if spatial.empty
                        else float(spatial.eq(MAX_EXTENT).mean() * 100)
                    ),
                    f"mean_{fit.predictor_output_name}": safe_mean(group[fit.predictor_col]),
                    f"median_{fit.predictor_output_name}": safe_median(group[fit.predictor_col]),
                    "age_used_median": source_fit.age_reference,
                    "sex_used_mean_male": source_fit.sex_reference,
                    "reference_APOE": source_fit.reference_apoe,
                    "model_scope": model_scope,
                    "data_source": fit.data_source,
                    **extra,
                }
            )

    return anchor_rows, full_grid_rows, llm_rows, curve_summary_rows, row_summary_rows


def generate_descriptives(
    base: PanelSample,
    ptau: PanelSample,
    max_days_ptau_pet: float,
) -> list[dict]:
    records: list[dict] = []
    configurations = [
        (base, "Abeta_burden_CL", "CL", {}),
        (
            ptau,
            "pTau217_zscore",
            "zscore_2sd",
            {"max_abs_days_diff_included": float(max_days_ptau_pet)},
        ),
    ]

    for sample, predictor_name, predictor_col, extra in configurations:
        for panel in PANEL_ORDER:
            panel_data = sample.panel(panel)
            for apoe in APOE_ORDER + ["ALL"]:
                subset = panel_data
                if apoe != "ALL":
                    subset = subset.loc[subset["apoe_key"].eq(apoe)]

                record = {
                    "DX": panel,
                    "panel": panel,
                    "panel_sample": PANEL_SAMPLE_RULE.get(panel, ""),
                    "APOE": apoe,
                    "model_predictor": predictor_name,
                    "n_subjects": int(len(subset)),
                    "data_source": sample.source,
                    **extra,
                }
                for column, prefix in [
                    (predictor_col, predictor_name),
                    ("spatial", "spatial_extent"),
                    ("AGE", "age"),
                ]:
                    stats = summary_block(subset[column])
                    for statistic, value in stats.items():
                        record[f"{prefix}_{statistic}"] = value

                spatial = pd.to_numeric(subset["spatial"], errors="coerce").dropna()
                record["pct_extent_ge_1"] = (
                    np.nan if spatial.empty else float(spatial.ge(1).mean() * 100)
                )
                record["pct_extent_ge_2"] = (
                    np.nan if spatial.empty else float(spatial.ge(2).mean() * 100)
                )
                record["max_extent"] = MAX_EXTENT
                record["pct_extent_eq_max"] = (
                    np.nan if spatial.empty else float(spatial.eq(MAX_EXTENT).mean() * 100)
                )
                records.append(record)

    return records


def save_dataframe(records: list[dict], path: Path) -> None:
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"Saved: {path} ({len(records):,} rows)")


def export_specification_results(output_dir: Path) -> None:
    terms: list[dict] = []
    summary: list[dict] = []
    for (predictor_output_name, panel), fits in SPECIFICATION_FITS.items():
        for spec in REPORTED_SPECS:
            fit = fits.get(spec)
            if fit is None:
                continue
            record = {
                "panel": panel,
                "model_predictor": predictor_output_name,
                "model_spec": spec,
                "model_spec_label": MODEL_SPECS[spec]["label"],
                "outcome": f"tau PET spatial extent (0-{MAX_EXTENT} regions)",
                "regions": " + ".join(POS_COLS),
                "apoe_sample": (
                    "APOE 3/3, 3/4, 4/4. Every epsilon-2 carrier (2/2, 2/3, 2/4) "
                    "is excluded from the dataset and appears in no panel"
                ),
                "model_status": fit.status,
                "model_message": fit.message,
                "used_for_figure_curves": bool(spec == CURVE_MODEL_SPEC),
                "n_subjects": int(len(fit.data)) if fit.data is not None else 0,
                "data_source": fit.data_source,
            }
            if fit.ok:
                params = fit.result.params
                errors = fit.result.bse
                pvalues = fit.result.pvalues
                conf = fit.result.conf_int()
                record["log_likelihood"] = float(fit.result.llf)
                for name in params.index:
                    if not isinstance(name, str) or "/" in str(name):
                        continue
                    terms.append({
                        "panel": panel,
                        "model_predictor": predictor_output_name,
                        "model_spec": spec,
                        "term": name,
                        "coefficient": float(params[name]),
                        "std_error": float(errors[name]),
                        "odds_ratio": float(np.exp(params[name])),
                        "or_ci_lower": float(np.exp(conf.loc[name, 0])),
                        "or_ci_upper": float(np.exp(conf.loc[name, 1])),
                        "p_value": float(pvalues[name]),
                        "n_subjects": int(len(fit.data)),
                        "used_for_figure_curves": bool(spec == CURVE_MODEL_SPEC),
                        "data_source": fit.data_source,
                    })
            summary.append(record)

    for records, filename in [
        (summary, "FIG_3_apoe_specification_summary.csv"),
        (terms, "FIG_3_apoe_specification_terms.csv"),
        (SPECIFICATION_TESTS, "FIG_3_apoe_specification_tests.csv"),
    ]:
        frame = pd.DataFrame(records)
        path = output_path(output_dir, filename)
        frame.to_csv(path, index=False)
        print(f"Saved: {path} ({len(frame):,} rows)")


def export_cell_audit(
    base: PanelSample,
    ptau: PanelSample,
    output_dir: Path,
) -> None:
    configurations = {
        1: (base, "CL_bin", "CL", "Abeta_burden_CL"),
        2: (ptau, "zscore_bin", "zscore_2sd", "pTau217_zscore"),
    }

    records: list[dict] = []
    for figure_row, panel, bin_label in AUDIT_CELLS:
        if figure_row not in configurations:
            raise ValueError(
                f"AUDIT_CELLS refers to figure row {figure_row}; only the "
                f"stacked rows {sorted(configurations)} can be audited."
            )
        if panel not in PANEL_ORDER:
            raise ValueError(f"AUDIT_CELLS refers to unknown panel '{panel}'.")

        sample, bin_column, predictor_col, predictor_name = configurations[figure_row]
        panel_data = sample.panel(panel)
        cell = panel_data.loc[
            panel_data[bin_column].astype("string").eq(bin_label)
        ].copy()

        if cell.empty:
            print(
                f"  [audit] {panel} × {bin_label} ({predictor_name}): no "
                "subjects. Check the bin label spelling against the configured "
                "bin labels."
            )
            continue

        spatial = pd.to_numeric(cell["spatial"], errors="coerce")
        counts = {extent: int(spatial.eq(extent).sum()) for extent in EXTENT_LEVELS}
        n_total = int(spatial.notna().sum())
        print(
            f"  [audit] {panel} × {bin_label} ({predictor_name}): n = {n_total}; "
            + ", ".join(f"extent {k}: {v}" for k, v in counts.items())
            + f"; {predictor_col} range "
            f"{safe_mean(cell[predictor_col]):.1f} mean / "
            f"[{pd.to_numeric(cell[predictor_col], errors='coerce').min():.1f}, "
            f"{pd.to_numeric(cell[predictor_col], errors='coerce').max():.1f}]"
        )

        for _, row in cell.iterrows():
            record = {
                "figure_row": figure_row,
                "panel": panel,
                "panel_sample": PANEL_SAMPLE_RULE.get(panel, ""),
                "x_variable": predictor_name,
                "x_bin_label": bin_label,
                "BIDS_SUBJECT": row.get("BIDS_SUBJECT"),
                "APOE": row.get("apoe_key"),
                "DX_LABEL": row.get("DX_LABEL"),
                "AGE": row.get("AGE"),
                "SEX_BINARY": row.get("SEX_BINARY"),
                predictor_col: row.get(predictor_col),
                "spatial_extent": row.get("spatial"),
                "data_source": sample.source,
            }
            record.update({column: row.get(column) for column in POS_COLS})
            if "abs_days_diff" in cell.columns:
                record["abs_days_diff"] = row.get("abs_days_diff")
            records.append(record)

    frame = pd.DataFrame(records)
    audit_path = output_path(output_dir, "FIG_3_cell_audit.csv")
    frame.to_csv(audit_path, index=False)
    print(f"Saved: {audit_path} ({len(frame):,} rows)")


def export_all_results(
    base: PanelSample,
    ptau: PanelSample,
    cl_models: dict[str, OrdinalFit],
    ptau_models: dict[str, OrdinalFit],
    output_dir: Path,
    max_days_ptau_pet: float,
) -> None:
    print("\nCreating quantitative exports...")

    export_cell_audit(base, ptau, output_dir)
    export_specification_results(output_dir)

    proportions, llm_stacked, row_stacked = generate_stacked_exports(
        base,
        ptau,
        max_days_ptau_pet,
    )

    all_fits = [cl_models[panel] for panel in PANEL_ORDER] + [
        ptau_models[panel] for panel in PANEL_ORDER
    ]
    model_wide, model_terms = model_result_rows(all_fits)
    anchors, full_grid, llm_models, curve_summary, row_models = (
        generate_model_prediction_exports(all_fits, max_days_ptau_pet)
    )
    descriptives = generate_descriptives(
        base,
        ptau,
        max_days_ptau_pet,
    )

    save_dataframe(
        model_wide, output_path(output_dir, "FIG_3_model_results.csv")
    )
    save_dataframe(
        proportions, output_path(output_dir, "FIG_3_stacked_proportions.csv")
    )
    save_dataframe(
        descriptives, output_path(output_dir, "FIG_3_descriptives.csv")
    )
    save_dataframe(
        anchors, output_path(output_dir, "FIG_3_expected_extent_anchors.csv")
    )
    save_dataframe(
        full_grid, output_path(output_dir, "FIG_3_expected_extent_full_grid.csv")
    )
    save_dataframe(
        llm_stacked + llm_models,
        output_path(output_dir, "FIG_3_LLM_subplot_summary.csv"),
    )
    save_dataframe(
        row_stacked + row_models,
        output_path(output_dir, "FIG_3_LLM_row_summary.csv"),
    )
    save_dataframe(
        curve_summary, output_path(output_dir, "FIG_3_LLM_curve_summary.csv")
    )
    save_dataframe(
        model_terms, output_path(output_dir, "FIG_3_LLM_model_terms.csv")
    )


def main() -> None:
    base_path = BASE_FILE.expanduser().resolve()
    ptau_path = PTAU_FILE.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")
    if not ptau_path.exists():
        raise FileNotFoundError(f"p-tau217 dataset not found: {ptau_path}")
    if MAX_DAYS_PTAU_PET < 0:
        raise ValueError("MAX_DAYS_PTAU_PET must be non-negative.")

    base, ptau = load_analysis_data(
        base_path,
        ptau_path,
        max_days_ptau_pet=MAX_DAYS_PTAU_PET,
    )

    cl_models, ptau_models = fit_all_models(base, ptau)

    figure_path = output_path(output_dir, FIGURE_NAME)
    build_figure(base, ptau, cl_models, ptau_models, figure_path)

    export_all_results(
        base,
        ptau,
        cl_models,
        ptau_models,
        output_dir,
        max_days_ptau_pet=MAX_DAYS_PTAU_PET,
    )

    print("\nFIG_3 analysis complete.")


if __name__ == "__main__":
    main()
