"""
Figure 6. Regional tau PET SUVR in APOE2/3 versus APOE3/3 Aβ-positive
participants: (a) [18F]flortaucipir and (b) [18F]MK-6240.

The script selects APOE 2/3 and 3/3 participants with Centiloid ≥ 12, excludes
co-pathology (COPAT = 1), and keeps the earliest eligible visit per participant
(Overall column) and per participant and diagnosis (CU, MCI and Dementia
columns). Centiloid and tau PET values come from input datasets in which the
amyloid and tau PET scans were already matched in time, so no additional
amyloid-to-tau PET interval filter is applied here. For each tracer, column and
region (Braak I–II, III–IV, V–VI and meta-ROI) it fits an ANCOVA (SUVR ~
APOE2/3 + age + sex + Centiloid + cohort, plus diagnosis in the Overall column)
with HC3 robust standard errors. Regional positivity and spatial extent are
analysed in the same way, with logistic and proportional-odds models as checks,
and Benjamini-Hochberg FDR is computed across all tests. The figure shows
covariate-adjusted SUVR as box plots with individual participants and the
ANCOVA p-value for each region. Sensitivity analyses without cohort adjustment,
over the full Centiloid range and without the CPAS cohort are also run.

Inputs:  DIMI_DATASET_07-09-2026.csv (analysis),
         DIMI_with_ptau217_07-09-2026.csv (consistency check of shared columns)
Outputs: FIG6_APOE23_vs_33_SUVR.png and CSV tables with ANCOVA results, figure
         values, adjusted points, descriptive statistics, demographics,
         sensitivity results and the input check.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy import stats as sps
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests


BASE_FILE = Path("DIMI_DATASET_07-09-2026.csv")
PTAU_FILE = Path("DIMI_with_ptau217_07-09-2026.csv")
OUTPUT_DIR = Path(".")
FIGURE_NAME = "FIG_6.png"

ANALYSIS_APOE = ["33", "23"]
REFERENCE_APOE = "33"
TEST_APOE = "23"
APOE_LABELS = {"33": "APOE3/3", "23": "APOE2/3"}

DX_ORDER = ["CU", "MCI", "Dementia"]
DX_MAP = {1: "CU", 2: "MCI", 3: "Dementia"}
OVERALL_LABEL = "Overall"
PANEL_ORDER = [OVERALL_LABEL, *DX_ORDER]
ADJUST_OVERALL_MODEL_FOR_DX = True

CL_THRESHOLD = 12
CL_SAMPLE = f"CL>={CL_THRESHOLD:g}"
CL_RULES = {
    CL_SAMPLE: lambda cl: cl >= CL_THRESHOLD,
    "All CL": lambda cl: cl.notna(),
}

TRACERS = {
    "Flortaucipir": {
        "pos": {"Braak 1-2": "POS_BRAAK12_COH", "Braak 3-4": "POS_BRAAK34_COH", "Braak 5-6": "POS_BRAAK56_COH"},
        "suvr": {"Braak 1-2": "braak12", "Braak 3-4": "braak34", "Braak 5-6": "braak56", "Meta-ROI": "metaroi"},
    },
    "MK-6240": {
        "pos": {"Braak 1-2": "MK_POS_BRAAK12_COH", "Braak 3-4": "MK_POS_BRAAK34_COH", "Braak 5-6": "MK_POS_BRAAK56_COH"},
        "suvr": {"Braak 1-2": "mk_braak12", "Braak 3-4": "mk_braak34", "Braak 5-6": "mk_braak56", "Meta-ROI": "mk_metaroi"},
    },
}
COHORT_COL = "COHORT"
EDUC_COL = "EDUC_YEARS"
COHORT_LABELS = {"A4": "A4/LEARN", "OASIS3": "OASIS-3"}

MIN_MODEL_N = 10
MIN_GROUP_N = 5
MIN_OUTCOME_LEVELS = 3

DATE_SORT_CANDIDATES = [
    "PET_DATE", "MK_DATE", "TAU_PET_DATE", "SCAN_DATE", "EXAM_DATE", "DATE",
    "SESSION_DATE", "VISIT_DATE", "AcquisitionDate", "acquisition_date",
]


@dataclass(frozen=True)
class Run:
    name: str
    adjust_for_cohort: bool
    cl_samples: tuple[str, ...]
    exclude_cohorts: tuple[str, ...] = ()


MAIN_RUN = Run(f"main: cohort-adjusted, {CL_SAMPLE}", adjust_for_cohort=True, cl_samples=(CL_SAMPLE,))

SENSITIVITY_RUNS = [
    Run(f"no cohort adjustment ({CL_SAMPLE} and full CL range)", adjust_for_cohort=False,
        cl_samples=(CL_SAMPLE, "All CL")),
    Run("without CPAS (no cohort adjustment)", adjust_for_cohort=False,
        cl_samples=(CL_SAMPLE, "All CL"), exclude_cohorts=("CPAS",)),
]


PALETTE_APOE = {"33": "#389F42", "23": "#2E86C1"}
TRACER_TEX = {"Flortaucipir": r"[$^{18}$F]flortaucipir", "MK-6240": r"[$^{18}$F]MK-6240"}
REGIONS = ["Braak 1-2", "Braak 3-4", "Braak 5-6", "Meta-ROI"]
REGION_TICK = {"Braak 1-2": "Braak\n1–2", "Braak 3-4": "Braak\n3–4", "Braak 5-6": "Braak\n5–6", "Meta-ROI": "Meta-\nROI"}

FONT_FAMILY = ["DejaVu Sans"]
FONT_SIZE = {"title": 20, "axis_label": 18, "tick": 14, "legend": 16, "annot": 12}
TITLE_BAND_PTS = 28

FIG_W = 20
COL_W, COL_GAP = 0.200, 0.06
ROW_H_IN, ROW_GAP_IN, TOP_IN, BOTTOM_IN = 4.0, 1.7, 0.8, 1.4
OFFSET = 0.19
BOX_W = 0.32
JITTER = 0.07
JITTER_SEED = 42
TEXT_GREY = "#444444"

Y_AXES = {
    ("Flortaucipir", "CU"):    ((0.75, 2.25), [1.0, 1.5, 2.0]),
    ("Flortaucipir", "other"): ((0.0, 3.5), [0.0, 1.0, 2.0, 3.0]),
    ("MK-6240", "CU"):         ((0.25, 3.25), [1.0, 2.0, 3.0]),
    ("MK-6240", "other"):      ((-0.3, 6.3), [0.0, 2.0, 4.0, 6.0]),
}


def normalize_apoe(values: pd.Series) -> pd.Series:
    raw = values.copy()
    result = pd.Series(pd.NA, index=raw.index, dtype="string")
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_mask = numeric.notna() & np.isclose(numeric, np.round(numeric))
    if numeric_mask.any():
        result.loc[numeric_mask] = np.round(numeric.loc[numeric_mask]).astype("Int64").astype("string")
    text_mask = result.isna() & raw.notna()
    if text_mask.any():
        cleaned = (
            raw.loc[text_mask].astype("string").str.lower()
            .str.replace("epsilon", "", regex=False).str.replace("apoe", "", regex=False)
            .str.replace("ε", "", regex=False).str.replace("e", "", regex=False)
            .str.replace(r"[^0-9]", "", regex=True)
        )
        result.loc[text_mask] = cleaned
    return result.replace({"43": "34", "32": "23", "42": "24"})


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
        mapped = pd.Series(pd.NA, index=cleaned.index, dtype="Float64")
        mapped.loc[cleaned.isin({"m", "male", "man", "masc", "masculino"})] = 1.0
        mapped.loc[cleaned.isin({"f", "female", "woman", "fem", "feminino"})] = 0.0
        result.loc[mapped.notna()] = mapped.loc[mapped.notna()]
    return result.astype(float)


def require_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def validate_binary_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    for column in columns:
        bad = df[column].notna() & ~df[column].isin([0, 1])
        if bad.any():
            raise ValueError(f"{dataset_name}: non-binary values in {column}.")


def choose_dedup_sort(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in DATE_SORT_CANDIDATES:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                return parsed, column
    if "AGE" in df.columns:
        return pd.to_numeric(df["AGE"], errors="coerce"), "AGE"
    return pd.Series(np.arange(len(df)), index=df.index), "row_order"


def deduplicate_first_visit(df: pd.DataFrame, dataset_name: str, keys: list[str], verbose: bool) -> pd.DataFrame:
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
    if verbose:
        print(f"    {dataset_name}: removed {before - len(work):,} longitudinal rows "
              f"within {' × '.join(keys)} using {sort_source}.")
    return work


def check_inputs(base: pd.DataFrame, ptau: pd.DataFrame) -> pd.DataFrame:
    key = ["BIDS_SUBJECT", "BIDS_SESSION"]
    used = ["APOE", "COPAT", "DX", "AGE", "SEX", "CL", COHORT_COL, EDUC_COL]
    for columns in TRACERS.values():
        used += [*columns["pos"].values(), *columns["suvr"].values()]
    a = base.drop_duplicates(key).set_index(key)
    b = ptau.drop_duplicates(key).set_index(key)
    rows = [{"check": "scans only in base file", "column": "", "n": len(a.index.difference(b.index))},
            {"check": "scans only in p-tau217 file", "column": "", "n": len(b.index.difference(a.index))}]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    for column in used:
        if column not in b.columns:
            rows.append({"check": "column missing in p-tau217 file", "column": column, "n": np.nan})
            continue
        x, y = a[column], b[column]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            same = np.isclose(x, y, rtol=0, atol=1e-9) | (x.isna() & y.isna())
        else:
            same = (x.astype(str) == y.astype(str)) | (x.isna() & y.isna())
        rows.append({"check": "scans with a different value", "column": column, "n": int((~same).sum())})
    report = pd.DataFrame(rows)
    problems = report.loc[report["n"].fillna(1).gt(0)]
    if problems.empty:
        print(f"  Input check: both files agree on all {len(used)} analysis columns for "
              f"{len(common):,} scans.")
    else:
        print("  Input check: WARNING, the two files differ:\n" + problems.to_string(index=False))
    return report


def prepare_tracer(df: pd.DataFrame, tracer: str, run: Run, dataset_name: str, verbose: bool) -> pd.DataFrame:
    columns = TRACERS[tracer]
    pos_cols = list(columns["pos"].values())
    suvr_cols = list(columns["suvr"].values())
    require_columns(df, ["BIDS_SUBJECT", "APOE", "COPAT", "DX", "AGE", "SEX", "CL", COHORT_COL,
                         *pos_cols, *suvr_cols], dataset_name)

    work = df.copy()
    work["APOE"] = normalize_apoe(work["APOE"])
    work["COPAT"] = pd.to_numeric(work["COPAT"], errors="coerce")
    work["DX"] = pd.to_numeric(work["DX"], errors="coerce")
    work["SEX_BINARY"] = normalize_sex(work["SEX"])
    for column in ["AGE", "CL", *pos_cols, *suvr_cols]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.loc[~work["COPAT"].eq(1)].copy()
    if run.exclude_cohorts:
        work = work.loc[~work[COHORT_COL].astype("string").isin(run.exclude_cohorts)].copy()

    work = work.loc[work["APOE"].isin(ANALYSIS_APOE) & work["DX"].isin(DX_MAP)].copy()
    work["DX_LABEL"] = work["DX"].map(DX_MAP)
    work["apoe_key"] = work["APOE"].astype("string")
    work["APOE_23"] = work["apoe_key"].eq(TEST_APOE).astype(float)

    validate_binary_columns(work, pos_cols, dataset_name)
    work["spatial"] = work[pos_cols].sum(axis=1, min_count=len(pos_cols))
    eligible = work.dropna(subset=["CL", "spatial", "AGE", "SEX_BINARY"]).copy()
    if verbose:
        print(f"  {tracer}: {len(eligible):,} eligible APOE3/3 or 2/3 scans (COPAT = 0, DX 1-3, "
              f"complete CL, age, sex and positivity).")
    return eligible


def build_samples(eligible: pd.DataFrame, tracer: str, run: Run, verbose: bool) -> dict[str, dict[str, pd.DataFrame]]:
    samples: dict[str, dict[str, pd.DataFrame]] = {}
    for sample_name in run.cl_samples:
        subset = eligible.loc[CL_RULES[sample_name](eligible["CL"])].copy()
        overall = deduplicate_first_visit(subset, f"{tracer} {sample_name} [Overall]", ["BIDS_SUBJECT"], verbose)
        by_dx = deduplicate_first_visit(subset, f"{tracer} {sample_name} [by DX]", ["BIDS_SUBJECT", "DX_LABEL"], verbose)
        panels = {OVERALL_LABEL: overall}
        for dx in DX_ORDER:
            panels[dx] = by_dx.loc[by_dx["DX_LABEL"].eq(dx)].copy()
        samples[sample_name] = panels
        if verbose:
            counts = pd.DataFrame({p: d["apoe_key"].value_counts().reindex(ANALYSIS_APOE, fill_value=0)
                                   for p, d in panels.items()}).T.rename(columns=APOE_LABELS)
            print(f"  {tracer} | {sample_name}: participants by panel\n{counts.to_string()}")
    return samples


def covariate_design(data: pd.DataFrame, panel: str, run: Run) -> pd.DataFrame:
    design = pd.DataFrame(index=data.index)
    design["APOE_23"] = data["APOE_23"]
    for column, name in [("AGE", "AGE_c"), ("SEX_BINARY", "SEX_c"), ("CL", "CL_c")]:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.nunique() > 1:
            design[name] = values - values.mean()
    if panel == OVERALL_LABEL and ADJUST_OVERALL_MODEL_FOR_DX:
        observed_dx = [dx for dx in DX_ORDER if data["DX_LABEL"].eq(dx).any()]
        for dx in observed_dx[1:]:
            dummy = data["DX_LABEL"].eq(dx).astype(float)
            design[f"DX_{dx}"] = dummy - dummy.mean()
    if run.adjust_for_cohort:
        counts = data[COHORT_COL].astype("string").value_counts()
        for cohort in counts.index[1:]:
            dummy = data[COHORT_COL].astype("string").eq(cohort).astype(float)
            design[f"COHORT_{cohort}"] = dummy - dummy.mean()
    return design.astype(float)


def support_problem(data: pd.DataFrame) -> str:
    n_ref = int(data["apoe_key"].eq(REFERENCE_APOE).sum())
    n_test = int(data["apoe_key"].eq(TEST_APOE).sum())
    if len(data) < MIN_MODEL_N:
        return f"n={len(data)} < {MIN_MODEL_N}"
    if min(n_ref, n_test) < MIN_GROUP_N:
        return f"APOE3/3 n={n_ref}, APOE2/3 n={n_test}; each needs ≥{MIN_GROUP_N}"
    return ""


def fit_ancova(data: pd.DataFrame, outcome: str, panel: str, run: Run) -> dict:
    exog = sm.add_constant(covariate_design(data, panel, run), has_constant="add")
    result = sm.OLS(data[outcome].astype(float), exog).fit(cov_type="HC3")
    params, cov = result.params, result.cov_params()
    contrast_23 = np.zeros(len(params))
    contrast_23[list(params.index).index("const")] = 1.0
    contrast_23[list(params.index).index("APOE_23")] = 1.0
    ci = result.conf_int().loc["APOE_23"]
    return {
        "adj_mean_APOE33": float(params["const"]),
        "adj_mean_APOE33_se": float(result.bse["const"]),
        "adj_mean_APOE23": float(contrast_23 @ params),
        "adj_mean_APOE23_se": float(np.sqrt(contrast_23 @ cov.values @ contrast_23)),
        "diff_23_minus_33": float(params["APOE_23"]),
        "diff_se": float(result.bse["APOE_23"]),
        "diff_ci_lo": float(ci.iloc[0]),
        "diff_ci_hi": float(ci.iloc[1]),
        "diff_p": float(result.pvalues["APOE_23"]),
        "coef_CL": float(params.get("CL_c", np.nan)),
        "r2": float(result.rsquared),
    }


def fit_check_model(data: pd.DataFrame, outcome: str, panel: str, kind: str, run: Run) -> dict:
    empty = {"check_model": kind, "check_OR": np.nan, "check_OR_ci_lo": np.nan,
             "check_OR_ci_hi": np.nan, "check_p": np.nan, "check_status": ""}
    design = covariate_design(data, panel, run)
    y = data[outcome].astype(int)
    if kind == "logistic":
        table = pd.crosstab(data["apoe_key"], y)
        if table.shape[1] < 2 or (table == 0).any().any():
            empty["check_status"] = f"not estimable (a genotype has 0 cases in one outcome level: {table.to_dict()})"
            return empty
        model = sm.Logit(y, sm.add_constant(design, has_constant="add"))
        fit_kwargs = {"disp": False, "maxiter": 200}
    else:
        if y.nunique() < MIN_OUTCOME_LEVELS:
            empty["check_status"] = f"only {y.nunique()} extent levels"
            return empty
        outcome_cat = pd.Categorical(y, categories=sorted(y.unique()), ordered=True)
        model = OrderedModel(outcome_cat, design, distr="logit")
        fit_kwargs = {"method": "bfgs", "maxiter": 500, "disp": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(**fit_kwargs)
        if not result.mle_retvals.get("converged", False):
            empty["check_status"] = "did not converge"
            return empty
        ci = result.conf_int().loc["APOE_23"]
        return {"check_model": kind, "check_OR": float(np.exp(result.params["APOE_23"])),
                "check_OR_ci_lo": float(np.exp(ci.iloc[0])), "check_OR_ci_hi": float(np.exp(ci.iloc[1])),
                "check_p": float(result.pvalues["APOE_23"]), "check_status": "ok"}
    except Exception as error:
        empty["check_status"] = f"{type(error).__name__}: {error}"
        return empty


def run_models(samples_by_tracer: dict, run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict] = []
    descriptives: list[dict] = []
    covariates = ("age, sex, CL" + (", diagnosis (Overall only)" if ADJUST_OVERALL_MODEL_FOR_DX else "")
                  + (", cohort" if run.adjust_for_cohort else ""))

    for tracer, samples in samples_by_tracer.items():
        columns = TRACERS[tracer]
        outcomes = ([("positivity", region, col) for region, col in columns["pos"].items()]
                    + [("spatial extent", "Braak 1-2 + 3-4 + 5-6", "spatial")]
                    + [("SUVR", region, col) for region, col in columns["suvr"].items()])
        for sample_name, panels in samples.items():
            for panel in PANEL_ORDER:
                panel_data = panels[panel]
                for apoe in ANALYSIS_APOE:
                    group = panel_data.loc[panel_data["apoe_key"].eq(apoe)]
                    record = {"tracer": tracer, "cl_sample": sample_name, "panel": panel,
                              "APOE": APOE_LABELS[apoe], "n": len(group),
                              "age_mean": group["AGE"].mean(), "age_sd": group["AGE"].std(),
                              "male_pct": group["SEX_BINARY"].mean() * 100,
                              "CL_mean": group["CL"].mean(), "CL_sd": group["CL"].std(),
                              "CL_median": group["CL"].median(),
                              "spatial_extent_mean": group["spatial"].mean()}
                    for region, col in columns["pos"].items():
                        record[f"pct_pos_{region}"] = group[col].mean() * 100
                    for region, col in columns["suvr"].items():
                        record[f"SUVR_mean_{region}"] = group[col].mean()
                    descriptives.append(record)

                for outcome_type, region, column in outcomes:
                    data = panel_data.dropna(subset=[column]).copy()
                    row = {"analysis": run.name, "tracer": tracer, "cl_sample": sample_name, "panel": panel,
                           "outcome_type": outcome_type, "region": region, "n_total": len(data),
                           "n_APOE33": int(data["apoe_key"].eq(REFERENCE_APOE).sum()),
                           "n_APOE23": int(data["apoe_key"].eq(TEST_APOE).sum()),
                           "covariates": covariates}
                    problem = support_problem(data)
                    if problem:
                        row["status"] = f"not fitted: {problem}"
                        results.append(row)
                        continue
                    try:
                        row.update(fit_ancova(data, column, panel, run))
                        row["status"] = "ok"
                    except Exception as error:
                        row["status"] = f"ANCOVA failed: {type(error).__name__}: {error}"
                        results.append(row)
                        continue
                    if outcome_type == "positivity":
                        row.update(fit_check_model(data, column, panel, "logistic", run))
                    elif outcome_type == "spatial extent":
                        row.update(fit_check_model(data, column, panel, "ordinal", run))
                    results.append(row)

    results_df = pd.DataFrame(results)
    fitted = results_df["status"].eq("ok")
    results_df["diff_q_FDR"] = np.nan
    if fitted.any():
        results_df.loc[fitted, "diff_q_FDR"] = multipletests(results_df.loc[fitted, "diff_p"], method="fdr_bh")[1]
    is_pos = results_df["outcome_type"].eq("positivity")
    for column in ["adj_mean_APOE33", "adj_mean_APOE23", "adj_mean_APOE33_se", "adj_mean_APOE23_se",
                   "diff_23_minus_33", "diff_se", "diff_ci_lo", "diff_ci_hi"]:
        if column in results_df:
            results_df.loc[is_pos, column] = results_df.loc[is_pos, column] * 100
    results_df["units"] = np.select(
        [is_pos, results_df["outcome_type"].eq("spatial extent")],
        ["percentage points", "number of positive regions (0-3)"], default="SUVR")
    return results_df, pd.DataFrame(descriptives)


def adjusted_suvr_points(samples_by_tracer: dict, run: Run) -> pd.DataFrame:
    records = []
    for tracer, samples in samples_by_tracer.items():
        for sample_name, panels in samples.items():
            for panel in PANEL_ORDER:
                for region, column in TRACERS[tracer]["suvr"].items():
                    data = panels[panel].dropna(subset=[column]).copy()
                    if support_problem(data):
                        continue
                    exog = sm.add_constant(covariate_design(data, panel, run), has_constant="add")
                    fit = sm.OLS(data[column].astype(float), exog).fit()
                    covariates = [c for c in exog.columns if c not in ("const", "APOE_23")]
                    adjusted = data[column].astype(float) - exog[covariates] @ fit.params[covariates]
                    for index in data.index:
                        records.append({"tracer": tracer, "cl_sample": sample_name, "panel": panel,
                                        "group": str(data.at[index, "apoe_key"]),
                                        "BIDS_SUBJECT": data.at[index, "BIDS_SUBJECT"],
                                        "BIDS_SESSION": data.at[index, "BIDS_SESSION"],
                                        "COHORT": data.at[index, COHORT_COL], "region": region,
                                        "suvr_raw": float(data.at[index, column]),
                                        "suvr_adjusted": float(adjusted.at[index])})
    return pd.DataFrame(records)


def run_analysis(raw: pd.DataFrame, run: Run, dataset_name: str, verbose: bool):
    samples = {}
    for tracer in TRACERS:
        eligible = prepare_tracer(raw, tracer, run, dataset_name, verbose)
        samples[tracer] = build_samples(eligible, tracer, run, verbose)
    results, descriptives = run_models(samples, run)
    return samples, results, descriptives


def figure_table(results: pd.DataFrame) -> pd.DataFrame:
    d = results.loc[results["outcome_type"].eq("SUVR")].copy()
    for group in ["APOE33", "APOE23"]:
        d[f"adj_mean_{group}_ci_lo"] = d[f"adj_mean_{group}"] - 1.96 * d[f"adj_mean_{group}_se"]
        d[f"adj_mean_{group}_ci_hi"] = d[f"adj_mean_{group}"] + 1.96 * d[f"adj_mean_{group}_se"]
    d["p_on_figure"] = d["diff_p"].map(format_p)
    d["_t"] = d["tracer"].map({t: i for i, t in enumerate(TRACERS)})
    d["_p"] = d["panel"].map({p: i for i, p in enumerate(PANEL_ORDER)})
    d["_r"] = d["region"].map({r: i for i, r in enumerate(REGIONS)})
    d = d.sort_values(["_t", "_p", "_r"])
    cols = ["tracer", "panel", "region", "n_APOE33", "n_APOE23",
            "adj_mean_APOE33", "adj_mean_APOE33_ci_lo", "adj_mean_APOE33_ci_hi",
            "adj_mean_APOE23", "adj_mean_APOE23_ci_lo", "adj_mean_APOE23_ci_hi",
            "diff_23_minus_33", "diff_ci_lo", "diff_ci_hi", "diff_p", "diff_q_FDR", "p_on_figure", "status"]
    return d[cols].reset_index(drop=True)


def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _mean_sd(values: pd.Series, decimals: int = 1) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return "—" if v.empty else f"{v.mean():.{decimals}f} ({v.std(ddof=1):.{decimals}f})"


def _median_iqr(values: pd.Series, decimals: int = 1) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return "—"
    return f"{v.median():.{decimals}f} [{v.quantile(0.25):.{decimals}f}, {v.quantile(0.75):.{decimals}f}]"


def _n_pct(mask: pd.Series) -> str:
    n, total = int(mask.sum()), int(mask.notna().sum())
    return "—" if total == 0 else f"{n} ({100 * n / total:.1f})"


def _welch_p(a: pd.Series, b: pd.Series) -> float:
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    return float(sps.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else np.nan


def _mwu_p(a: pd.Series, b: pd.Series) -> float:
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else np.nan


def _categorical_p(group: pd.Series, values: pd.Series) -> float:
    table = pd.crosstab(group, values)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    if table.shape == (2, 2):
        return float(sps.fisher_exact(table.values)[1])
    return float(sps.chi2_contingency(table.values)[1])


def demographic_rows(data: pd.DataFrame, tracer: str) -> list[dict]:
    columns = TRACERS[tracer]
    ref = data.loc[data["apoe_key"].eq(REFERENCE_APOE)]
    test = data.loc[data["apoe_key"].eq(TEST_APOE)]
    rows: list[dict] = []

    def add(section, variable, v_ref, v_test, p=np.nan, test_name=""):
        rows.append({"section": section, "variable": variable, "APOE3/3": v_ref, "APOE2/3": v_test,
                     "p": p, "p_text": _fmt_p(p) if np.isfinite(p) else "", "test": test_name})

    add("Participants", "n", str(len(ref)), str(len(test)))
    add("Participants", "Age, years, mean (SD)", _mean_sd(ref["AGE"]), _mean_sd(test["AGE"]),
        _welch_p(ref["AGE"], test["AGE"]), "Welch t")
    female = data["SEX_BINARY"].eq(0)
    add("Participants", "Female sex, n (%)", _n_pct(female.loc[ref.index]), _n_pct(female.loc[test.index]),
        _categorical_p(data["apoe_key"], female), "Fisher exact")
    if EDUC_COL in data:
        add("Participants", "Education, years, mean (SD)", _mean_sd(ref[EDUC_COL]), _mean_sd(test[EDUC_COL]),
            _welch_p(ref[EDUC_COL], test[EDUC_COL]), "Welch t")
    add("Clinical stage, n (%)", "Clinical stage, n (%)", "", "",
        _categorical_p(data["apoe_key"], data["DX_LABEL"]), "χ²")
    for dx, label in [("CU", "Cognitively unimpaired"), ("MCI", "MCI"), ("Dementia", "Dementia")]:
        add("Clinical stage, n (%)", f"  {label}", _n_pct(ref["DX_LABEL"].eq(dx)), _n_pct(test["DX_LABEL"].eq(dx)))
    add("Biomarkers", "Centiloid, median [IQR]", _median_iqr(ref["CL"]), _median_iqr(test["CL"]),
        _mwu_p(ref["CL"], test["CL"]), "Mann-Whitney U")
    for region, col in columns["pos"].items():
        positive = data[col].eq(1)
        add("Biomarkers", f"Tau PET positive, {region.replace('-', '–')}, n (%)", _n_pct(positive.loc[ref.index]),
            _n_pct(positive.loc[test.index]), _categorical_p(data["apoe_key"], positive), "Fisher exact")
    col = columns["suvr"]["Meta-ROI"]
    add("Biomarkers", "Tau PET Meta-ROI SUVR, mean (SD)", _mean_sd(ref[col], 2), _mean_sd(test[col], 2),
        _welch_p(ref[col], test[col]), "Welch t")
    cohort = data[COHORT_COL].astype("string")
    add("Contributing cohort, n (%)", "Contributing cohort, n (%)", "", "",
        _categorical_p(data["apoe_key"], cohort), "χ²")
    for name in sorted(cohort.dropna().unique()):
        add("Contributing cohort, n (%)", f"  {COHORT_LABELS.get(name, name)}",
            _n_pct(cohort.loc[ref.index].eq(name)), _n_pct(cohort.loc[test.index].eq(name)))
    return rows


def demographics_table(samples_by_tracer: dict) -> pd.DataFrame:
    records = []
    for tracer, samples in samples_by_tracer.items():
        for sample_name, panels in samples.items():
            for row in demographic_rows(panels[OVERALL_LABEL], tracer):
                records.append({"tracer": tracer, "cl_sample": sample_name, **row})
    return pd.DataFrame(records)


def format_p(p):
    if p is None or not np.isfinite(p):
        return ""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def style_ax(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_SIZE["tick"])
    ax.grid(False)


def add_title_with_box(ax, title):
    fig = ax.get_figure()
    band_h = TITLE_BAND_PTS / (ax.get_position().height * fig.get_size_inches()[1] * 72)
    ax.add_patch(Rectangle((0, 1.0), 1.0, band_h, transform=ax.transAxes, facecolor="#EBEBEB",
                           edgecolor="black", linewidth=plt.rcParams.get("axes.linewidth", 0.8),
                           clip_on=False, zorder=5))
    ax.text(0.5, 1.0 + band_h * 0.5, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_SIZE["title"], clip_on=False, zorder=6)


def fixed_axis(tracer: str, panel: str, values: pd.Series):
    limits, ticks = Y_AXES[(tracer, "CU" if panel == "CU" else "other")]
    v = values.dropna()
    if not v.empty and (v.min() < limits[0] or v.max() > limits[1]):
        print(f"  WARNING: {tracer} {panel}: data {v.min():.2f}–{v.max():.2f} outside fixed axis {limits}.")
    return limits, ticks


def apply_ticks(ax, ticks):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_locator(FixedLocator([]))


def draw_boxes(ax, points, stats, groups, ylim, title, rng):
    span = ylim[1] - ylim[0]
    drawn = False
    for x, region in enumerate(REGIONS):
        region_points = points.loc[points["region"].eq(region)]
        if region_points.empty:
            continue
        tops = []
        for offset, group in zip([-OFFSET, OFFSET], groups):
            values = region_points.loc[region_points["group"].eq(group), "suvr_adjusted"].dropna().to_numpy()
            if values.size == 0:
                continue
            color = PALETTE_APOE[group]
            ax.boxplot(values, positions=[x + offset], widths=BOX_W, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=matplotlib.colors.to_rgba(color, 0.22), edgecolor=color, linewidth=1.8),
                       medianprops=dict(color=color, linewidth=2.2),
                       whiskerprops=dict(color=color, linewidth=1.5), capprops=dict(color=color, linewidth=1.5),
                       zorder=2)
            alpha = 0.55 if values.size < 60 else 0.35
            size = 14 if values.size < 60 else 7
            ax.scatter(x + offset + rng.uniform(-JITTER, JITTER, values.size), values, s=size, color=color,
                       alpha=alpha, edgecolors="none", zorder=3)
            q1, q3 = np.percentile(values, [25, 75])
            tops.append(values[values <= q3 + 1.5 * (q3 - q1)].max())
            drawn = True
        hit = stats.loc[stats["region"].eq(region) & stats["status"].eq("ok")]
        if tops and not hit.empty:
            ax.text(x, min(max(tops) + 0.025 * span, ylim[1] - 0.085 * span), format_p(float(hit.iloc[0]["diff_p"])),
                    ha="center", va="bottom", fontsize=FONT_SIZE["annot"], color=TEXT_GREY, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))
    if not drawn:
        ax.text(0.5, 0.5, "Model unavailable\n(too few participants)", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=11)
    ax.set_xticks(range(len(REGIONS)))
    ax.set_xticklabels([REGION_TICK[r] for r in REGIONS], fontsize=FONT_SIZE["tick"] - 1)
    ax.set_xlim(-0.6, len(REGIONS) - 0.4)
    ax.set_ylim(*ylim)
    style_ax(ax)
    add_title_with_box(ax, title)


def make_figure(points: pd.DataFrame, results: pd.DataFrame, path_out: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
        "axes.titleweight": "regular", "font.size": FONT_SIZE["tick"], "axes.labelsize": FONT_SIZE["axis_label"],
        "xtick.labelsize": FONT_SIZE["tick"], "ytick.labelsize": FONT_SIZE["tick"],
    })
    rng = np.random.default_rng(JITTER_SEED)
    stats = results.loc[results["outcome_type"].eq("SUVR")]
    groups = [REFERENCE_APOE, TEST_APOE]
    tracers = list(TRACERS)

    n_rows = len(tracers)
    fig_h = TOP_IN + n_rows * ROW_H_IN + (n_rows - 1) * ROW_GAP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300, facecolor="white")
    total_w = len(PANEL_ORDER) * COL_W + (len(PANEL_ORDER) - 1) * COL_GAP
    start_x = (1.0 - total_w) / 2
    row_h = ROW_H_IN / fig_h
    bottoms = [(BOTTOM_IN + (n_rows - 1 - i) * (ROW_H_IN + ROW_GAP_IN)) / fig_h for i in range(n_rows)]
    label_x = max(0.005, start_x - 0.03)

    for i, tracer in enumerate(tracers):
        for j, panel in enumerate(PANEL_ORDER):
            ax = fig.add_axes([start_x + j * (COL_W + COL_GAP), bottoms[i], COL_W, row_h])
            p = points.loc[points["tracer"].eq(tracer) & points["panel"].eq(panel)]
            s = stats.loc[stats["tracer"].eq(tracer) & stats["panel"].eq(panel)]
            limits, ticks = fixed_axis(tracer, panel, p["suvr_adjusted"])
            draw_boxes(ax, p, s, groups, limits, panel, rng)
            apply_ticks(ax, ticks)
            ax.set_ylabel("Adjusted SUVR" if j == 0 else "", fontsize=FONT_SIZE["axis_label"], labelpad=6)
        y = bottoms[i] + row_h + 0.55 / fig_h
        fig.text(label_x, y, "ab"[i] + ".", ha="left", va="bottom", fontsize=FONT_SIZE["title"] + 1, fontweight="bold")
        fig.text(label_x + 0.0225, y, TRACER_TEX[tracer], ha="left", va="bottom", fontsize=FONT_SIZE["title"] + 1)

    width = 0.30
    lax = fig.add_axes([start_x + total_w / 2 - width / 2, bottoms[-1] - 1.05 / fig_h, width, 0.025])
    lax.axis("off")
    entries = [(r"$\it{APOE}$3/3", PALETTE_APOE["33"]), (r"$\it{APOE}$2/3", PALETTE_APOE["23"])]
    handles = [Patch(facecolor=matplotlib.colors.to_rgba(c, 0.22), edgecolor=c, linewidth=1.8, label=l)
               for l, c in entries]
    lax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=FONT_SIZE["legend"],
               handlelength=1.6, columnspacing=2.4)

    fig.savefig(path_out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path_out}")


def main() -> None:
    out = OUTPUT_DIR.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for path in (BASE_FILE, PTAU_FILE):
        if not path.expanduser().exists():
            raise FileNotFoundError(f"Input not found: {path.expanduser().resolve()}")

    print("Loading data...")
    base = pd.read_csv(BASE_FILE.expanduser(), low_memory=False)
    ptau = pd.read_csv(PTAU_FILE.expanduser(), low_memory=False)
    check_inputs(base, ptau).to_csv(out / "FIG6_input_check.csv", index=False)

    print(f"\nMain analysis ({MAIN_RUN.name})")
    samples, results, descriptives = run_analysis(base, MAIN_RUN, BASE_FILE.name, verbose=True)
    points = adjusted_suvr_points(samples, MAIN_RUN)
    table = figure_table(results)

    results.to_csv(out / "FIG6_ANCOVA_results.csv", index=False)
    table.to_csv(out / "FIG6_figure_table.csv", index=False)
    points.to_csv(out / "FIG6_adjusted_suvr_points.csv", index=False)
    descriptives.to_csv(out / "FIG6_descriptives.csv", index=False)
    demographics_table(samples).to_csv(out / "FIG6_demographics.csv", index=False)

    print("\nSensitivity analyses")
    sensitivity = []
    for run in SENSITIVITY_RUNS:
        _, res, _ = run_analysis(base, run, BASE_FILE.name, verbose=False)
        sensitivity.append(res)
        suvr = res.loc[res["outcome_type"].eq("SUVR") & res["status"].eq("ok")]
        top = suvr.sort_values("diff_p").iloc[0]
        print(f"  {run.name}: SUVR p < 0.05 in {int((suvr['diff_p'] < 0.05).sum())} of {len(suvr)}; "
              f"lowest SUVR q = {suvr['diff_q_FDR'].min():.2f} "
              f"(smallest p: {top['tracer']} {top['cl_sample']} {top['panel']} {top['region']}, "
              f"{top['diff_23_minus_33']:+.3f}, p = {top['diff_p']:.3f}, n APOE2/3 = {top['n_APOE23']})")
    pd.concat(sensitivity, ignore_index=True).to_csv(out / "FIG6_sensitivity_results.csv", index=False)

    make_figure(points, results, out / FIGURE_NAME)

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        grid = table.pivot_table(index=["tracer", "panel"], columns="region", values="p_on_figure",
                                 aggfunc="first", sort=False)
        print("\np-values printed on Figure 6:")
        print(grid.to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
