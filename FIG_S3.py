"""
Supplementary Figure S3. Difference in the expected number of tau-positive
regions between APOE ε4 carriers and APOE3/3 by Aβ burden and plasma p-tau217:
(a) [18F]flortaucipir and (b) [18F]MK-6240.

For each tracer, predictor (Centiloid or p-tau217 z-score) and column (Overall,
CU, MCI, Dementia), the script fits a proportional-odds ordinal logistic
regression of spatial extent (number of tau-positive Braak regions: I–II,
III–IV and V–VI; range 0–3) on the predictor, age, sex and APOE ε4 dose (3/3 =
0, 3/4 = 1, 4/4 = 2); the Overall model is also adjusted for diagnosis and its
curves are averaged over the observed diagnosis composition. It plots the
model-based difference in expected extent for APOE3/4 − 3/3 and APOE4/4 − 3/3
over the predictor range shared by the three genotypes, with pointwise (delta
method) and simultaneous (sup-t, 1,000 parametric bootstrap draws) 95%
confidence bands; the maximum of the APOE4/4 − 3/3 curve is marked. A model
adding a predictor × ε4 dose interaction is fitted as a likelihood-ratio check.
Participants with co-pathology (COPAT = 1) are excluded, p-tau217 must lie
within 365 days of the PET scan, and the earliest eligible visit is kept per
participant (Overall) and per participant and diagnosis. Centiloid and tau PET
values come from input datasets in which the amyloid and tau PET scans were
already matched in time, so no additional amyloid-to-tau PET interval filter is
applied here.

Inputs:  DIMI_with_ptau217_07-09-2026.csv
Outputs: folder outputs with FIG_S_TAUPET_APOE_difference.png/.pdf,
         tau_panel_summary.csv, tau_contrast_curves.csv,
         tau_model_coefficients.csv, tau_attrition.csv,
         tau_level_distribution.csv, run_manifest.json and run_log.txt.
         Optional settings: python FIG_S3.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

SCRIPT_NAME = "FIG_S_TAUPET_APOE_difference.py"
SCRIPT_VERSION = "2.0"

TAU_RAW_PATH = Path("DIMI_with_ptau217_07-09-2026.csv")
OUTPUT_DIR = Path("outputs")

APOE_ORDER = ["33", "34", "44"]
E4_DOSE = {"33": 0.0, "34": 1.0, "44": 2.0}
DX_MAP = {1: "CU", 2: "MCI", 3: "Dementia"}
DX_ORDER = ["CU", "MCI", "Dementia"]
PANEL_ORDER = ["Overall", *DX_ORDER]

POS_COLS = {
    "FLORTAUCIPIR": ["POS_BRAAK12_COH", "POS_BRAAK34_COH", "POS_BRAAK56_COH"],
    "MK6240": ["MK_POS_BRAAK12_COH", "MK_POS_BRAAK34_COH", "MK_POS_BRAAK56_COH"],
}
MAX_EXTENT = len(POS_COLS["FLORTAUCIPIR"])
EXTENT_LEVELS = list(range(MAX_EXTENT + 1))
ADJUST_OVERALL_MODEL_FOR_DX = True
TRACER_LABELS = {
    "FLORTAUCIPIR": r"a. [$^{18}$F]flortaucipir: difference in expected tau-positive regions",
    "MK6240": r"b. [$^{18}$F]MK-6240: difference in expected tau-positive regions",
}
TRACER_PLAIN = {
    "FLORTAUCIPIR": "[18F]flortaucipir",
    "MK6240": "[18F]MK-6240",
}

PREDICTORS = {
    "CL": {
        "label": "A-beta burden (CL)",
        "plot_label": r"A$\beta$ burden (CL)",
        "peak_symbol": "CL",
        "units": "Centiloid",
        "limits": (0.0, 200.0),
    },
    "zscore_2sd": {
        "label": "Plasma p-tau217 (z)",
        "plot_label": "Plasma p-tau217 (z)",
        "peak_symbol": "z",
        "units": "z-score (2 SD reference)",
        "limits": (0.0, 8.0),
    },
}
PREDICTOR_ORDER = ["CL", "zscore_2sd"]

MAX_DAYS_PTAU_PET = 365.0
MIN_GROUP_N = 5
MIN_PANEL_N = 40
N_GRID = 401
CI_LEVEL = 0.95
SIM_DRAWS = 1000
RANDOM_SEED = 20240517

COLOR_44 = "#C7353E"
COLOR_34 = "#E0B04A"

TITLE_BAND_PTS = 28
TITLE_BAND_FACECOLOR = "#EBEBEB"
TITLE_FONT_SIZE = 12.5
LEGEND_FONT_SIZE = 10.5

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "black",
    }
)


class RunLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def warn(self, message: str) -> None:
        self.log(f"[WARNING] {message}")

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


LOG = RunLogger()


class Attrition:
    def __init__(self) -> None:
        self.records: list[dict] = []

    @staticmethod
    def _counts(frame: pd.DataFrame) -> tuple[int, int]:
        rows = int(len(frame))
        if "BIDS_SUBJECT" in frame.columns:
            subjects = int(frame["BIDS_SUBJECT"].nunique(dropna=True))
        else:
            subjects = -1
        return rows, subjects

    def record(
        self,
        scope: str,
        step_order: int,
        step: str,
        rule: str,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ) -> None:
        rows_before, subj_before = self._counts(before)
        rows_after, subj_after = self._counts(after)
        self.records.append(
            {
                "scope": scope,
                "step_order": step_order,
                "step": step,
                "rule_applied": rule,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
                "subjects_before": subj_before,
                "subjects_after": subj_after,
                "subjects_removed": subj_before - subj_after,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "scope",
                    "step_order",
                    "step",
                    "rule_applied",
                    "rows_before",
                    "rows_after",
                    "rows_removed",
                    "subjects_before",
                    "subjects_after",
                    "subjects_removed",
                ]
            )
        return pd.DataFrame(self.records)


ATTRITION = Attrition()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_id(tracer: str, predictor: str, dx: str) -> str:
    return f"{tracer}__{predictor}__{dx}"


def style_spines(ax) -> None:
    for side in ("top", "right", "bottom", "left"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_linewidth(plt.rcParams["axes.linewidth"])
        spine.set_edgecolor(plt.rcParams["axes.edgecolor"])


def add_title_with_box(ax, title: str, fontsize: float = TITLE_FONT_SIZE) -> None:
    fig = ax.get_figure()
    ax_height_pts = ax.get_position().height * fig.get_size_inches()[1] * 72.0
    band_h = TITLE_BAND_PTS / max(ax_height_pts, 1e-9)
    ax.add_patch(
        Rectangle(
            xy=(0.0, 1.0),
            width=1.0,
            height=band_h,
            transform=ax.transAxes,
            facecolor=TITLE_BAND_FACECOLOR,
            edgecolor=plt.rcParams["axes.edgecolor"],
            linewidth=plt.rcParams["axes.linewidth"],
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
        fontsize=fontsize,
        clip_on=False,
        zorder=6,
    )


def add_bottom_legend(fig, handles, y: float, width: float = 0.74) -> None:
    n_items = len(handles)
    ncol = n_items if n_items <= 3 else (2 if n_items <= 4 else 3)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.2, y, width, 0.08),
        borderaxespad=0.0,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.4,
        columnspacing=1.4,
        handletextpad=0.6,
        labelspacing=0.8,
        alignment="center",
    )


def normalize_apoe(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    mask = numeric.notna() & np.isclose(numeric, np.round(numeric))
    result.loc[mask] = np.round(numeric.loc[mask]).astype("Int64").astype("string")
    return result.replace({"43": "34"})


def normalize_sex(values: pd.Series) -> pd.Series:
    raw = values.copy()
    result = pd.Series(pd.NA, index=raw.index, dtype="Float64")

    numeric = pd.to_numeric(raw, errors="coerce")
    present = numeric.dropna()
    if not present.empty:
        observed = {float(v) for v in np.unique(present)}
        if observed.issubset({0.0, 1.0}):
            result.loc[numeric.notna()] = numeric.loc[numeric.notna()].astype(float)
        elif observed.issubset({1.0, 2.0}):
            mapped = numeric.map({1.0: 1.0, 2.0: 0.0})
            result.loc[mapped.notna()] = mapped.loc[mapped.notna()]
        else:
            raise ValueError(
                "SEX column has an unrecognised numeric coding "
                f"(observed values: {sorted(observed)}). Expected {{0,1}} or {{1,2}}. "
                "Recode the column or run with --sex-adjustment none and document it."
            )

    text_mask = result.isna() & raw.notna()
    if text_mask.any():
        cleaned = raw.loc[text_mask].astype("string").str.strip().str.lower()
        male_tokens = {"m", "male", "man", "masc", "masculino"}
        female_tokens = {"f", "female", "woman", "fem", "feminino"}
        mapped = pd.Series(pd.NA, index=cleaned.index, dtype="Float64")
        mapped.loc[cleaned.isin(male_tokens)] = 1.0
        mapped.loc[cleaned.isin(female_tokens)] = 0.0
        unresolved = cleaned.loc[mapped.isna()].unique().tolist()
        if unresolved:
            raise ValueError(
                "SEX column has unrecognised text values: "
                f"{unresolved[:10]}. Recode them or run with --sex-adjustment none."
            )
        result.loc[mapped.notna()] = mapped.loc[mapped.notna()]

    return result.astype(float)


def validate_columns(frame: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required column(s): {', '.join(missing)}. "
            f"Columns present: {', '.join(map(str, frame.columns[:60]))}"
        )


def prepare(
    raw: pd.DataFrame,
    tracer: str,
    pos_cols: list[str],
    sex_adjustment: str,
) -> pd.DataFrame:
    scope = f"{tracer}::prepare"
    work = raw.copy()

    base_required = ["BIDS_SUBJECT", "APOE", "DX", "AGE", "CL", "zscore_2sd", "days_diff"]
    validate_columns(work, base_required, "Tau CSV")
    validate_columns(work, pos_cols, f"Tau CSV (tracer {tracer})")

    ATTRITION.record(scope, 0, "input rows", "none (raw file as read)", work, work)

    if sex_adjustment == "none":
        work["SEX_BINARY"] = np.nan
        LOG.log(f"  {tracer}: sex adjustment disabled by --sex-adjustment none.")
    elif "SEX" in work.columns:
        work["SEX_BINARY"] = normalize_sex(work["SEX"])
    elif sex_adjustment == "require":
        raise ValueError(
            "SEX column is absent but the model specifies a sex term. "
            "Provide SEX, or run with --sex-adjustment none (and say so in the methods)."
        )
    else:
        work["SEX_BINARY"] = np.nan
        LOG.warn(f"{tracer}: SEX column absent; proceeding without sex adjustment.")

    numeric = ["AGE", "CL", "zscore_2sd", "days_diff", "DX", "SEX_BINARY", *pos_cols]
    if "COPAT" in work.columns:
        numeric.append("COPAT")
    else:
        work["COPAT"] = 0.0
        numeric.append("COPAT")
        LOG.log(f"  {tracer}: COPAT column absent; all rows treated as COPAT = 0.")
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work["APOE"] = normalize_apoe(work["APOE"])

    before = work
    work = work.loc[~work["COPAT"].eq(1)].copy()
    ATTRITION.record(scope, 1, "co-pathology filter", "exclude COPAT == 1 (missing COPAT kept)", before, work)

    before = work
    work = work.loc[work["APOE"].isin(APOE_ORDER)].copy()
    ATTRITION.record(
        scope, 2, "APOE genotype filter", "keep 3/3, 3/4 (43 recoded to 34), 4/4", before, work
    )

    before = work
    work = work.loc[work["DX"].isin(DX_MAP)].copy()
    ATTRITION.record(scope, 3, "diagnosis filter", "keep DX in {1:CU, 2:MCI, 3:Dementia}", before, work)

    work["DX_LABEL"] = work["DX"].map(DX_MAP)
    work["apoe_key"] = work["APOE"].astype("string")
    work["E4_DOSE"] = work["apoe_key"].map(E4_DOSE).astype(float)
    work["spatial"] = work[pos_cols].sum(axis=1, min_count=len(pos_cols))
    work["abs_days_diff"] = work["days_diff"].abs()

    before = work
    work = work.loc[work["spatial"].notna()].copy()
    ATTRITION.record(
        scope,
        4,
        "complete tau positivity",
        f"all {len(pos_cols)} regional positivity flags non-missing ({', '.join(pos_cols)})",
        before,
        work,
    )

    before = work
    work = work.loc[work["spatial"].isin(EXTENT_LEVELS)].copy()
    ATTRITION.record(scope, 5, "valid extent range", f"spatial extent in 0-{MAX_EXTENT}", before, work)

    work["spatial"] = work["spatial"].astype(int)
    return work


def dedup_visit(df: pd.DataFrame, keys: list[str], selection: str) -> pd.DataFrame:
    work = df.copy()
    work["_SORT"] = pd.to_numeric(work["AGE"], errors="coerce")
    work["_ORDER"] = np.arange(len(work))
    keep = "last" if selection == "latest" else "first"
    return (
        work.sort_values([*keys, "_SORT", "_ORDER"], na_position="last" if selection == "earliest" else "first")
        .drop_duplicates(keys, keep=keep)
        .drop(columns=["_SORT", "_ORDER"])
        .reset_index(drop=True)
    )


def split_panels(eligible: pd.DataFrame, selection: str, scope: str) -> dict[str, pd.DataFrame]:
    overall = dedup_visit(eligible, ["BIDS_SUBJECT"], selection)
    ATTRITION.record(
        scope,
        20,
        "cross-sectional reduction (Overall)",
        f"one {selection} eligible visit per BIDS_SUBJECT (ordered by AGE)",
        eligible,
        overall,
    )
    by_dx = dedup_visit(eligible, ["BIDS_SUBJECT", "DX_LABEL"], selection)
    ATTRITION.record(
        scope,
        21,
        "cross-sectional reduction (stage panels)",
        f"one {selection} eligible visit per BIDS_SUBJECT per DX_LABEL (ordered by AGE)",
        eligible,
        by_dx,
    )
    panels = {"Overall": overall}
    for dx in DX_ORDER:
        panels[dx] = by_dx.loc[by_dx["DX_LABEL"].eq(dx)].reset_index(drop=True)
    return panels


def build_eligible_sets(
    prepared: pd.DataFrame,
    tracer: str,
    max_days: float,
    selection: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}

    scope = f"{tracer}::CL"
    before = prepared
    cl_eligible = prepared.dropna(subset=["CL", "spatial", "AGE"]).copy()
    ATTRITION.record(
        scope, 10, "predictor eligibility", "non-missing CL, spatial extent and AGE", before, cl_eligible
    )
    out["CL"] = split_panels(cl_eligible, selection, scope)

    scope = f"{tracer}::zscore_2sd"
    before = prepared
    window = prepared.loc[prepared["abs_days_diff"].le(max_days)].copy()
    ATTRITION.record(
        scope,
        10,
        "p-tau to PET temporal window",
        f"absolute interval between plasma p-tau217 and tau PET <= {max_days:g} days",
        before,
        window,
    )
    before = window
    ptau_eligible = window.dropna(subset=["zscore_2sd", "spatial", "AGE"]).copy()
    ATTRITION.record(
        scope,
        11,
        "predictor eligibility",
        "non-missing plasma p-tau217 z-score, spatial extent and AGE",
        before,
        ptau_eligible,
    )
    out["zscore_2sd"] = split_panels(ptau_eligible, selection, scope)

    return out


def group_support(
    data: pd.DataFrame, column: str, limits: tuple[float, float] | None, min_group_n: int
) -> tuple[float, float] | None:
    lowers, uppers = [], []
    for apoe in APOE_ORDER:
        values = pd.to_numeric(data.loc[data["apoe_key"].eq(apoe), column], errors="coerce").dropna()
        if len(values) < min_group_n:
            return None
        lowers.append(float(values.min()))
        uppers.append(float(values.max()))
    lower, upper = max(lowers), min(uppers)
    if limits is not None:
        lower, upper = max(lower, limits[0]), min(upper, limits[1])
    return (lower, upper) if upper > lower else None


def _delta_se(fun, params: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray(fun(params), dtype=float)
    jacobian = np.zeros((len(estimate), len(params)))
    for index in range(len(params)):
        step = 1e-5 * max(1.0, abs(params[index]))
        plus, minus = params.copy(), params.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (np.asarray(fun(plus)) - np.asarray(fun(minus))) / (2.0 * step)
    variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
    return estimate, np.sqrt(np.clip(variance, 0.0, None))


def _mvn_draws(mean: np.ndarray, cov: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    symmetric = (cov + cov.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, 0.0, None)
    root = vectors * np.sqrt(values)
    return mean + rng.standard_normal((n_draws, len(mean))) @ root.T


@dataclass
class PanelResult:
    tracer: str
    predictor: str
    dx: str
    status: str
    status_detail: str = ""
    x: np.ndarray | None = None
    difference: dict = field(default_factory=dict)
    se: dict = field(default_factory=dict)
    lower_pointwise: dict = field(default_factory=dict)
    upper_pointwise: dict = field(default_factory=dict)
    lower_simultaneous: dict = field(default_factory=dict)
    upper_simultaneous: dict = field(default_factory=dict)
    support: tuple[float, float] | None = None
    group_n: dict = field(default_factory=dict)
    group_subjects: dict = field(default_factory=dict)
    level_counts: dict = field(default_factory=dict)
    coefficients: pd.DataFrame | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def pid(self) -> str:
        return panel_id(self.tracer, self.predictor, self.dx)


def fit_panel(
    data: pd.DataFrame,
    tracer: str,
    predictor: str,
    dx: str,
    limits: tuple[float, float],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> PanelResult:
    result_shell = PanelResult(tracer=tracer, predictor=predictor, dx=dx, status="not_fitted")

    sex_usable = (
        args.sex_adjustment != "none"
        and "SEX_BINARY" in data.columns
        and bool(data["SEX_BINARY"].notna().any())
    )
    needed = [predictor, "AGE", "spatial", "apoe_key", "E4_DOSE"]
    if sex_usable:
        needed.append("SEX_BINARY")
    subset = data.dropna(subset=needed).copy()

    result_shell.group_n = {a: int(subset["apoe_key"].eq(a).sum()) for a in APOE_ORDER}
    result_shell.group_subjects = {
        a: int(subset.loc[subset["apoe_key"].eq(a), "BIDS_SUBJECT"].nunique()) for a in APOE_ORDER
    }
    result_shell.level_counts = {
        int(level): int((subset["spatial"] == level).sum()) for level in EXTENT_LEVELS
    }

    if len(subset) < args.min_panel_n:
        result_shell.status = "skipped"
        result_shell.status_detail = (
            f"panel n = {len(subset)} below --min-panel-n = {args.min_panel_n}"
        )
        return result_shell

    smallest = min(result_shell.group_n.values())
    if smallest < args.min_group_n:
        result_shell.status = "skipped"
        result_shell.status_detail = (
            f"smallest genotype group n = {smallest} below --min-group-n = {args.min_group_n}"
        )
        return result_shell

    if subset["spatial"].nunique() < 2:
        result_shell.status = "skipped"
        result_shell.status_detail = "outcome has fewer than two distinct extent levels"
        return result_shell

    support = group_support(subset, predictor, limits, args.min_group_n)
    if support is None:
        result_shell.status = "skipped"
        result_shell.status_detail = (
            "no common predictor support across the three genotypes within the plotted limits"
        )
        return result_shell
    result_shell.support = support

    levels = np.sort(subset["spatial"].unique()).astype(int)
    age_reference = float(subset["AGE"].median())

    design = pd.DataFrame(index=subset.index)
    design[predictor] = subset[predictor].astype(float)

    age_centered = subset["AGE"].astype(float) - age_reference
    age_in_model = age_centered.nunique(dropna=True) > 1
    if age_in_model:
        design["AGE_centered"] = age_centered

    sex_reference = np.nan
    sex_in_model = False
    if sex_usable:
        sex_values = pd.to_numeric(subset["SEX_BINARY"], errors="coerce").astype(float)
        sex_reference = float(sex_values.mean())
        if sex_values.nunique(dropna=True) > 1:
            design["SEX_centered"] = sex_values - sex_reference
            sex_in_model = True

    dose = subset["E4_DOSE"].astype(float)
    if dose.nunique(dropna=True) < 2:
        result_shell.status = "skipped"
        result_shell.status_detail = "APOE e4 dose is constant within the panel"
        return result_shell
    design["E4_DOSE"] = dose
    dx_weights: dict[str, float] = {}
    if dx == "Overall" and ADJUST_OVERALL_MODEL_FOR_DX:
        observed_dx = [stage for stage in DX_ORDER if subset["DX_LABEL"].eq(stage).any()]
        if len(observed_dx) >= 2:
            for stage in observed_dx[1:]:
                design[f"DX_{stage}"] = subset["DX_LABEL"].eq(stage).astype(float)
            dx_weights = {stage: float(subset["DX_LABEL"].eq(stage).mean()) for stage in observed_dx}
    design = design.astype(float)

    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        result_shell.status = "failed"
        result_shell.status_detail = "design matrix is rank deficient"
        return result_shell

    outcome = pd.Categorical(subset["spatial"], categories=levels.tolist(), ordered=True)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = OrderedModel(outcome, design, distr="logit")
            fit = model.fit(method="bfgs", maxiter=1000, disp=False)
        fit_warnings = "; ".join(sorted({str(w.message) for w in caught}))
    except Exception as exc:
        result_shell.status = "failed"
        result_shell.status_detail = f"{type(exc).__name__}: {exc}"
        LOG.warn(f"{result_shell.pid}: model failed -- {type(exc).__name__}: {exc}")
        return result_shell

    converged = bool(fit.mle_retvals.get("converged", True))
    params = np.asarray(fit.params, dtype=float)
    covariance = np.asarray(fit.cov_params(), dtype=float)
    if not converged or not np.all(np.isfinite(params)) or not np.all(np.isfinite(covariance)):
        result_shell.status = "failed"
        result_shell.status_detail = "optimizer did not converge or produced non-finite estimates"
        LOG.warn(f"{result_shell.pid}: {result_shell.status_detail}")
        return result_shell

    x = np.linspace(support[0], support[1], args.n_grid)

    def prediction_block(apoe: str, stage: str | None = None) -> np.ndarray:
        block = pd.DataFrame(0.0, index=np.arange(len(x)), columns=design.columns)
        block[predictor] = x
        if "AGE_centered" in block.columns:
            block["AGE_centered"] = 0.0
        if "SEX_centered" in block.columns:
            block["SEX_centered"] = 0.0
        block["E4_DOSE"] = E4_DOSE[apoe]
        if stage is not None and f"DX_{stage}" in block.columns:
            block[f"DX_{stage}"] = 1.0
        return block.to_numpy(float)

    if dx_weights:
        blocks = {
            a: [(weight, prediction_block(a, stage)) for stage, weight in dx_weights.items()]
            for a in APOE_ORDER
        }
    else:
        blocks = {a: [(1.0, prediction_block(a))] for a in APOE_ORDER}
    weights = levels.astype(float)

    def expected(theta: np.ndarray, weighted_blocks: list) -> np.ndarray:
        total = 0.0
        for share, block in weighted_blocks:
            probabilities = np.asarray(model.predict(theta, exog=block), dtype=float)
            total = total + share * (probabilities @ weights)
        return total

    critical = float(stats.norm.ppf(0.5 + args.ci_level / 2.0))

    difference, se_map = {}, {}
    for apoe in ("34", "44"):
        estimate, se = _delta_se(
            lambda theta, a=apoe: expected(theta, blocks[a]) - expected(theta, blocks["33"]),
            params,
            covariance,
        )
        difference[apoe] = estimate
        se_map[apoe] = se

    lower_pw = {a: difference[a] - critical * se_map[a] for a in difference}
    upper_pw = {a: difference[a] + critical * se_map[a] for a in difference}

    lower_sim, upper_sim, sim_critical = {}, {}, {}
    if args.simultaneous_draws > 0:
        draws = _mvn_draws(params, covariance, args.simultaneous_draws, rng)
        for apoe in ("34", "44"):
            curves = np.vstack(
                [expected(theta, blocks[apoe]) - expected(theta, blocks["33"]) for theta in draws]
            )
            safe_se = np.where(se_map[apoe] > 0, se_map[apoe], np.nan)
            zeta = np.nanmax(np.abs((curves - difference[apoe]) / safe_se), axis=1)
            crit_sim = float(np.nanquantile(zeta, args.ci_level))
            sim_critical[apoe] = crit_sim
            lower_sim[apoe] = difference[apoe] - crit_sim * se_map[apoe]
            upper_sim[apoe] = difference[apoe] + crit_sim * se_map[apoe]

    names = list(fit.params.index) if hasattr(fit.params, "index") else [f"p{i}" for i in range(len(params))]
    conf = np.asarray(fit.conf_int(alpha=1 - args.ci_level), dtype=float)
    term_type = ["threshold" if "/" in str(n) else "regression coefficient" for n in names]
    coefficients = pd.DataFrame(
        {
            "panel_id": result_shell.pid,
            "term": names,
            "term_type": term_type,
            "estimate_log_odds": params,
            "std_error": np.asarray(fit.bse, dtype=float),
            "z": np.asarray(fit.tvalues, dtype=float),
            "p_value": np.asarray(fit.pvalues, dtype=float),
            "ci_low": conf[:, 0],
            "ci_high": conf[:, 1],
            "odds_ratio_per_unit": np.exp(params),
        }
    )
    coefficients.loc[coefficients["term_type"].eq("threshold"), "odds_ratio_per_unit"] = np.nan

    coefficients["threshold_cutpoint"] = np.nan
    n_beta = design.shape[1]
    try:
        cutpoints = np.asarray(
            model.transform_threshold_params(params[n_beta:]), dtype=float
        )
        interior = cutpoints[np.isfinite(cutpoints)]
        threshold_rows = coefficients.index[coefficients["term_type"].eq("threshold")]
        if len(interior) == len(threshold_rows):
            coefficients.loc[threshold_rows, "threshold_cutpoint"] = interior
    except Exception:
        pass

    interaction = {
        "lr_stat": np.nan,
        "lr_df": 1,
        "lr_p": np.nan,
        "beta": np.nan,
        "se": np.nan,
        "status": "not_run",
    }
    if args.interaction_test:
        try:
            design_full = design.copy()
            design_full["PREDICTOR_x_E4_DOSE"] = design[predictor] * design["E4_DOSE"]
            if np.linalg.matrix_rank(design_full.to_numpy(float)) == design_full.shape[1]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    full_fit = OrderedModel(outcome, design_full, distr="logit").fit(
                        method="bfgs", maxiter=1000, disp=False
                    )
                if bool(full_fit.mle_retvals.get("converged", True)):
                    stat = 2.0 * (float(full_fit.llf) - float(fit.llf))
                    interaction = {
                        "lr_stat": stat,
                        "lr_df": 1,
                        "lr_p": float(stats.chi2.sf(max(stat, 0.0), 1)),
                        "beta": float(full_fit.params["PREDICTOR_x_E4_DOSE"]),
                        "se": float(full_fit.bse["PREDICTOR_x_E4_DOSE"]),
                        "status": "fitted",
                    }
                else:
                    interaction["status"] = "interaction model did not converge"
            else:
                interaction["status"] = "interaction design rank deficient"
        except Exception as exc:
            interaction["status"] = f"{type(exc).__name__}: {exc}"

    result_shell.status = "fitted"
    result_shell.status_detail = fit_warnings
    result_shell.x = x
    result_shell.difference = difference
    result_shell.se = se_map
    result_shell.lower_pointwise = lower_pw
    result_shell.upper_pointwise = upper_pw
    result_shell.lower_simultaneous = lower_sim
    result_shell.upper_simultaneous = upper_sim
    result_shell.coefficients = coefficients
    result_shell.diagnostics = {
        "n_observations": int(len(subset)),
        "n_subjects": int(subset["BIDS_SUBJECT"].nunique()),
        "age_reference_median": age_reference,
        "sex_reference_mean_male": sex_reference,
        "age_term_in_model": age_in_model,
        "sex_term_in_model": sex_in_model,
        "diagnosis_term_in_model": bool(dx_weights),
        "dx_marginalization_weights": "; ".join(f"{stage}={share:.6f}" for stage, share in dx_weights.items()),
        "levels_observed": ",".join(str(int(v)) for v in levels),
        "n_levels_observed": int(len(levels)),
        "expected_extent_ceiling": float(weights.max()),
        "log_likelihood": float(fit.llf),
        "aic": float(fit.aic),
        "converged": converged,
        "fit_warnings": fit_warnings,
        "interaction": interaction,
        "sim_critical_34": sim_critical.get("34", np.nan),
        "sim_critical_44": sim_critical.get("44", np.nan),
        "predictor_observed_min": float(subset[predictor].min()),
        "predictor_observed_max": float(subset[predictor].max()),
    }
    return result_shell


def draw_panel(
    ax,
    result: PanelResult,
    xlabel: str,
    ylabel: str | None,
    title: str | None,
    ylim: tuple[float, float],
    show_yticklabels: bool,
    xlim: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    style_spines(ax)
    ax.grid(False)
    ax.set_axisbelow(True)
    if title:
        add_title_with_box(ax, title, TITLE_FONT_SIZE)
    ax.set_xlabel(xlabel, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=6)

    if result.status != "fitted":
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(
            f"not estimable\n({result.status})",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=9.5,
            style="italic",
            color="0.45",
        )
        return

    ax.axhline(0.0, color="0.45", lw=0.9, zorder=1)

    for apoe, color, width in (("34", COLOR_34, 2.2), ("44", COLOR_44, 2.4)):
        ax.fill_between(
            result.x,
            result.lower_pointwise[apoe],
            result.upper_pointwise[apoe],
            color=color,
            alpha=0.16,
            lw=0,
            zorder=3,
        )
        if args.simultaneous_draws > 0 and apoe in result.lower_simultaneous:
            ax.plot(result.x, result.lower_simultaneous[apoe], color=color, lw=0.9, ls=(0, (3, 2)), zorder=4)
            ax.plot(result.x, result.upper_simultaneous[apoe], color=color, lw=0.9, ls=(0, (3, 2)), zorder=4)
        ax.plot(result.x, result.difference[apoe], color=color, lw=width, zorder=5)

    if args.show_peak:
        peak = int(np.argmax(result.difference["44"]))
        interior = 0 < peak < len(result.x) - 1
        ax.plot(
            [result.x[peak]],
            [result.difference["44"][peak]],
            marker="o",
            ms=6,
            mfc="white",
            mec=COLOR_44,
            mew=1.8,
            zorder=6,
            clip_on=False,
        )
        span = result.support[1] - result.support[0]
        symbol = PREDICTORS[result.predictor]["peak_symbol"]
        value = f"{result.x[peak]:.0f}" if span > 20 else f"{result.x[peak]:.1f}"
        label = f"{symbol}={value}"
        alignment = "left" if peak == 0 else ("right" if not interior else "center")
        dx_offset = 4 if peak == 0 else (-4 if not interior else 0)
        peak_fraction = (result.difference["44"][peak] - ylim[0]) / max(ylim[1] - ylim[0], 1e-9)
        above = peak_fraction < 0.90
        ax.annotate(
            label + ("" if interior else "*"),
            xy=(result.x[peak], result.difference["44"][peak]),
            xytext=(dx_offset, 9 if above else -14),
            textcoords="offset points",
            ha=alignment,
            va="bottom" if above else "top",
            fontsize=9.5,
            color="black",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="none", edgecolor="none"),
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if not show_yticklabels:
        ax.set_yticklabels([])


def build_figure(results: dict[str, PanelResult], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    tracers = args.tracers_resolved
    fig = plt.figure(figsize=(16, 12) if len(tracers) == 2 else (16, 6.4))

    if len(tracers) == 2:
        block_extent = {0: (0.920, 0.565), 1: (0.462, 0.118)}
        block_label_y = {0: 0.968, 1: 0.510}
    else:
        block_extent = {0: (0.860, 0.215)}
        block_label_y = {0: 0.950}

    for block_index, tracer in enumerate(tracers):
        top, bottom = block_extent[block_index]
        grid = fig.add_gridspec(
            2, 4, top=top, bottom=bottom, left=0.062, right=0.985, hspace=0.62, wspace=0.10
        )
        for row, predictor in enumerate(PREDICTOR_ORDER):
            meta = PREDICTORS[predictor]
            for col, dx in enumerate(PANEL_ORDER):
                ax = fig.add_subplot(grid[row, col])
                draw_panel(
                    ax,
                    results[panel_id(tracer, predictor, dx)],
                    meta["plot_label"],
                    "Difference in expected\ntau-positive regions" if col == 0 else None,
                    dx,
                    args.ylim_resolved,
                    col == 0,
                    meta["limits"],
                    args,
                )
        fig.text(
            0.062,
            block_label_y[block_index],
            TRACER_LABELS[tracer],
            fontsize=14.5,
            ha="left",
            va="center",
        )

    handles = [
        Line2D([], [], color=COLOR_44, lw=2.4, label=r"$\it{APOE}$4/4 − $\it{APOE}$3/3"),
        Line2D([], [], color=COLOR_34, lw=2.2, label=r"$\it{APOE}$3/4 − $\it{APOE}$3/3"),
        Patch(facecolor=COLOR_44, alpha=0.16, label=f"{args.ci_level:.0%} pointwise confidence band"),
    ]
    if args.simultaneous_draws > 0:
        handles.append(
            Line2D(
                [],
                [],
                color=COLOR_44,
                lw=0.9,
                ls=(0, (3, 2)),
                label=f"{args.ci_level:.0%} simultaneous band (sup-t over plotted range)",
            )
        )
    if args.show_peak:
        handles.append(
            Line2D(
                [],
                [],
                color=COLOR_44,
                marker="o",
                ls="none",
                ms=6,
                mfc="white",
                mew=1.8,
                label="Descriptive maximum of 4/4 − 3/3 (* = at edge of common support)",
            )
        )

    add_bottom_legend(fig, handles, y=0.012)

    stem = output_dir / "FIG_S3"
    paths = []
    for suffix, kwargs in ((".png", {"dpi": 250}), (".pdf", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def curves_frame(results: dict[str, PanelResult]) -> pd.DataFrame:
    rows = []
    for result in results.values():
        if result.status != "fitted":
            continue
        for apoe in ("34", "44"):
            frame = pd.DataFrame(
                {
                    "panel_id": result.pid,
                    "tracer": result.tracer,
                    "predictor": result.predictor,
                    "predictor_units": PREDICTORS[result.predictor]["units"],
                    "diagnosis_panel": result.dx,
                    "contrast": f"APOE{apoe[0]}/{apoe[1]} minus APOE3/3",
                    "e4_dose_compared": E4_DOSE[apoe],
                    "predictor_value": result.x,
                    "difference_expected_regions": result.difference[apoe],
                    "std_error": result.se[apoe],
                    "ci_low_pointwise": result.lower_pointwise[apoe],
                    "ci_high_pointwise": result.upper_pointwise[apoe],
                }
            )
            if apoe in result.lower_simultaneous:
                frame["ci_low_simultaneous"] = result.lower_simultaneous[apoe]
                frame["ci_high_simultaneous"] = result.upper_simultaneous[apoe]
            else:
                frame["ci_low_simultaneous"] = np.nan
                frame["ci_high_simultaneous"] = np.nan
            frame["excludes_zero_pointwise"] = (
                (frame["ci_low_pointwise"] > 0) | (frame["ci_high_pointwise"] < 0)
            ).astype(int)
            frame["excludes_zero_simultaneous"] = (
                (frame["ci_low_simultaneous"] > 0) | (frame["ci_high_simultaneous"] < 0)
            ).astype("Int64")
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summary_frame(results: dict[str, PanelResult], args: argparse.Namespace) -> pd.DataFrame:
    records = []
    for result in results.values():
        record = {
            "panel_id": result.pid,
            "tracer": result.tracer,
            "tracer_label": TRACER_PLAIN[result.tracer],
            "predictor": result.predictor,
            "predictor_units": PREDICTORS[result.predictor]["units"],
            "diagnosis_panel": result.dx,
            "status": result.status,
            "status_detail": result.status_detail,
            "n_observations": result.diagnostics.get("n_observations", np.nan),
            "n_subjects": result.diagnostics.get("n_subjects", np.nan),
            "n_APOE33": result.group_n.get("33", 0),
            "n_APOE34": result.group_n.get("34", 0),
            "n_APOE44": result.group_n.get("44", 0),
            "subjects_APOE33": result.group_subjects.get("33", 0),
            "subjects_APOE34": result.group_subjects.get("34", 0),
            "subjects_APOE44": result.group_subjects.get("44", 0),
        }
        for level in EXTENT_LEVELS:
            record[f"n_extent_{level}"] = result.level_counts.get(level, 0)

        record["support_low"] = result.support[0] if result.support else np.nan
        record["support_high"] = result.support[1] if result.support else np.nan
        record["model_formula"] = (
            "ordinal logit: extent ~ predictor + age_centered + sex_centered + e4_dose"
            + (" + diagnosis" if result.diagnostics.get("diagnosis_term_in_model", False) else "")
        )
        record["model_has_interaction"] = 0
        record["age_term_in_model"] = int(bool(result.diagnostics.get("age_term_in_model", False)))
        record["sex_term_in_model"] = int(bool(result.diagnostics.get("sex_term_in_model", False)))
        record["diagnosis_term_in_model"] = int(bool(result.diagnostics.get("diagnosis_term_in_model", False)))
        record["dx_marginalization_weights"] = result.diagnostics.get("dx_marginalization_weights", "")
        record["age_reference_median"] = result.diagnostics.get("age_reference_median", np.nan)
        record["sex_reference_mean_male"] = result.diagnostics.get("sex_reference_mean_male", np.nan)
        record["levels_observed"] = result.diagnostics.get("levels_observed", "")
        record["expected_extent_ceiling"] = result.diagnostics.get("expected_extent_ceiling", np.nan)
        record["log_likelihood"] = result.diagnostics.get("log_likelihood", np.nan)
        record["aic"] = result.diagnostics.get("aic", np.nan)
        record["converged"] = int(bool(result.diagnostics.get("converged", False)))

        interaction = result.diagnostics.get("interaction", {})
        record["interaction_lr_stat"] = interaction.get("lr_stat", np.nan)
        record["interaction_lr_df"] = interaction.get("lr_df", np.nan)
        record["interaction_lr_p"] = interaction.get("lr_p", np.nan)
        record["interaction_beta_log_odds"] = interaction.get("beta", np.nan)
        record["interaction_se"] = interaction.get("se", np.nan)
        record["interaction_test_status"] = interaction.get("status", "not_run")

        if result.status == "fitted":
            for apoe in ("34", "44"):
                diff = result.difference[apoe]
                arg_max = int(np.argmax(diff))
                arg_min = int(np.argmin(diff))
                prefix = f"contrast_{apoe}"
                record[f"{prefix}_max_difference"] = float(diff[arg_max])
                record[f"{prefix}_max_at_predictor"] = float(result.x[arg_max])
                record[f"{prefix}_max_is_interior"] = int(0 < arg_max < len(result.x) - 1)
                record[f"{prefix}_max_ci_low_pointwise"] = float(result.lower_pointwise[apoe][arg_max])
                record[f"{prefix}_max_ci_high_pointwise"] = float(result.upper_pointwise[apoe][arg_max])
                if apoe in result.lower_simultaneous:
                    record[f"{prefix}_max_ci_low_simultaneous"] = float(
                        result.lower_simultaneous[apoe][arg_max]
                    )
                    record[f"{prefix}_max_ci_high_simultaneous"] = float(
                        result.upper_simultaneous[apoe][arg_max]
                    )
                else:
                    record[f"{prefix}_max_ci_low_simultaneous"] = np.nan
                    record[f"{prefix}_max_ci_high_simultaneous"] = np.nan
                record[f"{prefix}_min_difference"] = float(diff[arg_min])
                record[f"{prefix}_min_at_predictor"] = float(result.x[arg_min])
                record[f"{prefix}_at_support_low"] = float(diff[0])
                record[f"{prefix}_at_support_high"] = float(diff[-1])
                pointwise_significant = (
                    (result.lower_pointwise[apoe] > 0) | (result.upper_pointwise[apoe] < 0)
                )
                record[f"{prefix}_fraction_grid_pointwise_excludes_zero"] = float(
                    pointwise_significant.mean()
                )
                if apoe in result.lower_simultaneous:
                    simultaneous_significant = (
                        (result.lower_simultaneous[apoe] > 0) | (result.upper_simultaneous[apoe] < 0)
                    )
                    record[f"{prefix}_fraction_grid_simultaneous_excludes_zero"] = float(
                        simultaneous_significant.mean()
                    )
                else:
                    record[f"{prefix}_fraction_grid_simultaneous_excludes_zero"] = np.nan
                record[f"{prefix}_simultaneous_critical_value"] = result.diagnostics.get(
                    f"sim_critical_{apoe}", np.nan
                )
        record["max_is_descriptive_only"] = 1
        record["ci_level"] = args.ci_level
        records.append(record)
    return pd.DataFrame(records)


def level_distribution_frame(results: dict[str, PanelResult]) -> pd.DataFrame:
    records = []
    for result in results.values():
        for level, count in sorted(result.level_counts.items()):
            records.append(
                {
                    "panel_id": result.pid,
                    "tracer": result.tracer,
                    "predictor": result.predictor,
                    "diagnosis_panel": result.dx,
                    "extent_level": level,
                    "n_observations": count,
                    "level_observed_in_model": int(
                        str(level) in str(result.diagnostics.get("levels_observed", "")).split(",")
                    ),
                }
            )
    return pd.DataFrame(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tau-PET APOE genotype-difference supplementary figure. Fits an additive "
            "proportional-odds ordinal model of tau-positive spatial extent and writes a "
            "fully documented set of outputs."
        )
    )
    parser.add_argument("--tau-csv", type=Path, default=TAU_RAW_PATH, help=f"Tau-PET analysis CSV (default: {TAU_RAW_PATH})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument(
        "--tracers",
        choices=["all", "flortaucipir", "mk6240"],
        default="all",
        help="Which tracer block(s) to analyse.",
    )
    parser.add_argument(
        "--visit-selection",
        choices=["latest", "earliest"],
        default="earliest",
        help="Which eligible visit to retain per subject (ordered by age).",
    )
    parser.add_argument(
        "--max-days-ptau-pet",
        type=float,
        default=MAX_DAYS_PTAU_PET,
        help="Maximum absolute plasma p-tau217 to tau-PET interval, in days.",
    )
    parser.add_argument(
        "--sex-adjustment",
        choices=["require", "optional", "none"],
        default="require",
        help=(
            "require: fail if SEX is absent (default, matches the stated model). "
            "optional: proceed without the sex term and warn. "
            "none: deliberately fit without sex."
        ),
    )
    parser.add_argument("--min-group-n", type=int, default=MIN_GROUP_N, help="Minimum observations per genotype in a panel.")
    parser.add_argument("--min-panel-n", type=int, default=MIN_PANEL_N, help="Minimum total observations in a panel.")
    parser.add_argument("--n-grid", type=int, default=N_GRID, help="Number of evaluation points along the predictor.")
    parser.add_argument("--ci-level", type=float, default=CI_LEVEL, help="Nominal coverage for all intervals.")
    parser.add_argument(
        "--simultaneous-draws",
        type=int,
        default=SIM_DRAWS,
        help="Parametric-bootstrap draws for sup-t simultaneous bands. 0 disables them.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for the simultaneous-band calibration.")
    parser.add_argument(
        "--ylim",
        default="-1,2.5",
        help="Shared y-axis limits as 'low,high', or 'auto' to derive them from the fitted curves.",
    )
    parser.add_argument(
        "--no-peak",
        dest="show_peak",
        action="store_false",
        help="Hide the descriptive maximum marker. Recommended when the additive model is used.",
    )
    parser.set_defaults(show_peak=True)
    parser.add_argument(
        "--no-interaction-test",
        dest="interaction_test",
        action="store_false",
        help="Skip the predictor-by-dose likelihood-ratio specification check.",
    )
    parser.set_defaults(interaction_test=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status if any panel fails to fit.",
    )
    return parser.parse_args(argv)


def resolve_ylim(argument: str, results: dict[str, PanelResult]) -> tuple[float, float]:
    if argument.strip().lower() != "auto":
        low, high = (float(v) for v in argument.split(","))
        return (low, high)
    values: list[float] = []
    for result in results.values():
        if result.status != "fitted":
            continue
        for apoe in ("34", "44"):
            values.extend([float(result.lower_pointwise[apoe].min()), float(result.upper_pointwise[apoe].max())])
    if not values:
        return (-1.0, 1.0)
    low, high = min(values), max(values)
    pad = 0.10 * (high - low)
    return (min(low - pad, 0.0), high + pad)


def main(argv: list[str] | None = None) -> int:
    started = datetime.now(timezone.utc)
    args = parse_args(argv)

    tracer_choice = {
        "all": ["FLORTAUCIPIR", "MK6240"],
        "flortaucipir": ["FLORTAUCIPIR"],
        "mk6240": ["MK6240"],
    }[args.tracers]
    args.tracers_resolved = tracer_choice

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.log(f"{SCRIPT_NAME} v{SCRIPT_VERSION}")
    LOG.log(f"  started (UTC): {started.isoformat()}")
    if not args.tau_csv.exists():
        raise FileNotFoundError(f"Tau CSV not found: {args.tau_csv}")
    input_hash = sha256_of(args.tau_csv)
    LOG.log(f"  input: {args.tau_csv} (sha256 {input_hash[:16]}...)")
    LOG.log("  model: extent ~ predictor + age_centered + sex_centered + e4_dose (+ diagnosis in Overall; additive, no interaction)")

    raw = pd.read_csv(args.tau_csv)
    LOG.log(f"  read {len(raw):,} rows x {raw.shape[1]} columns")

    rng = np.random.default_rng(args.seed)
    results: dict[str, PanelResult] = {}

    for tracer in tracer_choice:
        LOG.log(f"  --- {tracer} ({TRACER_PLAIN[tracer]}) ---")
        prepared = prepare(raw, tracer, POS_COLS[tracer], args.sex_adjustment)
        eligible = build_eligible_sets(prepared, tracer, args.max_days_ptau_pet, args.visit_selection)
        for predictor in PREDICTOR_ORDER:
            for dx in PANEL_ORDER:
                result = fit_panel(
                    eligible[predictor][dx],
                    tracer,
                    predictor,
                    dx,
                    PREDICTORS[predictor]["limits"],
                    args,
                    rng,
                )
                results[result.pid] = result
                if result.status == "fitted":
                    LOG.log(
                        f"    {result.pid}: fitted, n = {result.diagnostics['n_observations']:,} "
                        f"(3/3 {result.group_n['33']}, 3/4 {result.group_n['34']}, 4/4 {result.group_n['44']}), "
                        f"support {result.support[0]:.1f} to {result.support[1]:.1f}"
                    )
                else:
                    LOG.log(f"    {result.pid}: {result.status} -- {result.status_detail}")

    args.ylim_resolved = resolve_ylim(args.ylim, results)

    figure_paths = build_figure(results, args, output_dir)
    LOG.log(f"  wrote figure: {', '.join(p.name for p in figure_paths)}")

    summary = summary_frame(results, args)
    curves = curves_frame(results)
    coefficients = pd.concat(
        [r.coefficients for r in results.values() if r.coefficients is not None], ignore_index=True
    ) if any(r.coefficients is not None for r in results.values()) else pd.DataFrame()
    attrition = ATTRITION.to_frame()
    levels = level_distribution_frame(results)

    written = {
        "tau_panel_summary.csv": summary,
        "tau_contrast_curves.csv": curves,
        "tau_model_coefficients.csv": coefficients,
        "tau_attrition.csv": attrition,
        "tau_level_distribution.csv": levels,
    }
    for name, frame in written.items():
        frame.to_csv(output_dir / name, index=False)
        LOG.log(f"  wrote {name} ({len(frame):,} rows)")

    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "run_started_utc": started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join([SCRIPT_NAME, *(argv if argv is not None else sys.argv[1:])]),
        "input_file": {
            "path": str(args.tau_csv.resolve()),
            "sha256": input_hash,
            "rows_read": int(len(raw)),
            "columns_read": int(raw.shape[1]),
        },
        "output_dir": str(output_dir.resolve()),
        "model": {
            "family": "proportional-odds ordinal logistic (logit link)",
            "formula": "extent ~ predictor + age_centered + sex_centered + e4_dose (+ diagnosis in the Overall panel)",
            "interaction_in_fitted_model": False,
            "interaction_sensitivity_test": bool(args.interaction_test),
            "reference_genotype": "APOE 3/3 (e4 dose 0)",
            "outcome": "number of tau-positive Braak regions (I-II, III-IV, V-VI) out of three (0-3, ordinal)",
        },
        "settings": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k not in {"ylim_resolved", "tracers_resolved"}
        },
        "resolved": {
            "tracers": tracer_choice,
            "y_axis_limits": list(args.ylim_resolved),
            "panels_attempted": int(len(results)),
            "panels_fitted": int(sum(r.status == "fitted" for r in results.values())),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "files_written": [p.name for p in figure_paths]
        + list(written.keys())
        + ["run_manifest.json", "run_log.txt"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOG.log("  wrote run_manifest.json")

    failures = [r for r in results.values() if r.status == "failed"]
    if failures:
        LOG.warn(f"{len(failures)} panel(s) failed to fit; see tau_panel_summary.csv status_detail.")

    LOG.save(output_dir / "run_log.txt")

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
