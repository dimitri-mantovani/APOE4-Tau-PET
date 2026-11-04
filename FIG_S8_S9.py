"""
Supplementary Figures S8 and S9. Regional tau PET SUVR in women versus men
within APOE haplotypes (APOE3/3, 3/4 and 4/4) among Aβ-positive participants,
one figure for [18F]flortaucipir and one for [18F]MK-6240.

The script selects APOE 3/3, 3/4 and 4/4 participants with Centiloid ≥ 12,
excludes co-pathology (COPAT = 1), and keeps the earliest eligible visit per
participant (Overall column) and per participant and diagnosis (CU, MCI and
Dementia columns). Centiloid and tau PET values come from input datasets in
which the amyloid and tau PET scans were already matched in time, so no
additional amyloid-to-tau PET interval filter is applied here. Within each
haplotype, tracer, column and region (Braak I–II, III–IV, V–VI and meta-ROI) it
fits an ANCOVA (SUVR ~ female sex + age + Centiloid + cohort, plus diagnosis in
the Overall column) with HC3 robust standard errors. Regional positivity and
spatial extent are analysed in the same way, with logistic and proportional-
odds models as checks. Whether the sex difference varies by haplotype is tested
with a sex × APOE interaction model (joint robust Wald test). Benjamini-
Hochberg FDR is computed across the sex tests and, separately, across the
interaction tests. Each figure shows covariate-adjusted SUVR as box plots with
individual participants and the ANCOVA p-value for each region.

Inputs:  DIMI_DATASET_07-09-2026.csv (analysis),
         DIMI_with_ptau217_07-09-2026.csv (consistency check of shared columns)
Outputs: SUPP_SUVR_sex_within_APOE_Flortaucipir.png,
         SUPP_SUVR_sex_within_APOE_MK6240.png and CSV tables with results,
         interaction tests, counts, adjusted points, figure values and the
         input check.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy.stats import chi2 as sps_chi2
from scipy.stats import norm as sps_norm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests


BASE_FILE = Path("DIMI_DATASET_07-09-2026.csv")
PTAU_FILE = Path("DIMI_with_ptau217_07-09-2026.csv")
OUTPUT_DIR = Path(".")
PREFIX = "SUPP_sex_within_APOE"

ANALYSIS_APOE = ["33", "34", "44"]
INTERACTION_REFERENCE_APOE = "33"
APOE_SHORT = {"23": "2/3", "33": "3/3", "34": "3/4", "44": "4/4"}

DX_ORDER = ["CU", "MCI", "Dementia"]
DX_MAP = {1: "CU", 2: "MCI", 3: "Dementia"}
OVERALL_LABEL = "Overall"
PANEL_ORDER = [OVERALL_LABEL, *DX_ORDER]
ADJUST_OVERALL_MODEL_FOR_DX = True
ADJUST_FOR_COHORT = True

CL_THRESHOLD = 12
SAMPLE_NAME = f"CL>={CL_THRESHOLD:g}"

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

MIN_MODEL_N = 10
MIN_GROUP_N = 5
MIN_OUTCOME_LEVELS = 3
LEVERAGE_ONE_TOL = 1e-8

DATE_SORT_CANDIDATES = [
    "PET_DATE", "MK_DATE", "TAU_PET_DATE", "SCAN_DATE", "EXAM_DATE", "DATE",
    "SESSION_DATE", "VISIT_DATE", "AcquisitionDate", "acquisition_date",
]


PALETTE_SEX = {"men": "#4E6E81", "women": "#A9447B"}
TRACER_SLUG = {"Flortaucipir": "Flortaucipir", "MK-6240": "MK6240"}
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
TEXT_GREY = "#444444"

JITTER_SEED = 42
JITTER_SKIP = 8044

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


def deduplicate_first_visit(df: pd.DataFrame, dataset_name: str, keys: list[str]) -> pd.DataFrame:
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
    print(f"    {dataset_name}: removed {before - len(work):,} longitudinal rows within "
          f"{' × '.join(keys)} using {sort_source}.")
    return work


def check_inputs(base: pd.DataFrame, ptau: pd.DataFrame) -> pd.DataFrame:
    key = ["BIDS_SUBJECT", "BIDS_SESSION"]
    used = ["APOE", "COPAT", "DX", "AGE", "SEX", "CL", COHORT_COL]
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
        print(f"  Input check: both files agree on all {len(used)} analysis columns for {len(common):,} scans.")
    else:
        print("  Input check: WARNING, the two files differ:\n" + problems.to_string(index=False))
    return report


def prepare_tracer(df: pd.DataFrame, tracer: str, dataset_name: str) -> pd.DataFrame:
    columns = TRACERS[tracer]
    pos_cols, suvr_cols = list(columns["pos"].values()), list(columns["suvr"].values())
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
    genotyped = work.loc[work["APOE"].notna() & work["DX"].isin(DX_MAP)]
    dropped = genotyped.loc[~genotyped["APOE"].isin(ANALYSIS_APOE), "APOE"]
    print(f"  {tracer}: kept APOE {', '.join(APOE_SHORT[a] for a in ANALYSIS_APOE)}; dropped "
          f"{len(dropped):,} rows {dropped.value_counts().to_dict()}.")

    work = work.loc[work["APOE"].isin(ANALYSIS_APOE) & work["DX"].isin(DX_MAP)].copy()
    work["DX_LABEL"] = work["DX"].map(DX_MAP)
    work["apoe_key"] = work["APOE"].astype("string")
    work["FEMALE"] = 1.0 - work["SEX_BINARY"]

    validate_binary_columns(work, pos_cols, dataset_name)
    work["spatial"] = work[pos_cols].sum(axis=1, min_count=len(pos_cols))
    return work.dropna(subset=["CL", "spatial", "AGE", "SEX_BINARY"]).copy()


def build_panels(eligible: pd.DataFrame, tracer: str) -> dict[str, pd.DataFrame]:
    subset = eligible.loc[eligible["CL"].ge(CL_THRESHOLD)].copy()
    print(f"  {tracer} | {SAMPLE_NAME}: {len(subset):,} eligible rows before deduplication")
    overall = deduplicate_first_visit(subset, f"{tracer} [Overall]", ["BIDS_SUBJECT"])
    by_dx = deduplicate_first_visit(subset, f"{tracer} [by DX]", ["BIDS_SUBJECT", "DX_LABEL"])
    panels = {OVERALL_LABEL: overall, **{dx: by_dx.loc[by_dx["DX_LABEL"].eq(dx)].copy() for dx in DX_ORDER}}
    table = pd.concat(
        {p: d.assign(sex=np.where(d["FEMALE"].eq(1), "women", "men")).groupby(["apoe_key", "sex"]).size()
         for p, d in panels.items()}, axis=1).fillna(0).astype(int)
    print(f"    n by APOE × sex and panel:\n{table.to_string()}")
    return panels


def adjustment_columns(data: pd.DataFrame, panel: str) -> pd.DataFrame:
    design = pd.DataFrame(index=data.index)
    for column, name in [("AGE", "AGE_c"), ("CL", "CL_c")]:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.nunique() > 1:
            design[name] = values - values.mean()
    if panel == OVERALL_LABEL and ADJUST_OVERALL_MODEL_FOR_DX:
        observed_dx = [dx for dx in DX_ORDER if data["DX_LABEL"].eq(dx).any()]
        for dx in observed_dx[1:]:
            dummy = data["DX_LABEL"].eq(dx).astype(float)
            design[f"DX_{dx}"] = dummy - dummy.mean()
    if ADJUST_FOR_COHORT:
        counts = data[COHORT_COL].astype("string").value_counts()
        for cohort in counts.index[1:]:
            dummy = data[COHORT_COL].astype("string").eq(cohort).astype(float)
            design[f"COHORT_{cohort}"] = dummy - dummy.mean()
    return design.astype(float)


def sex_design(data: pd.DataFrame, panel: str) -> pd.DataFrame:
    return pd.concat([data[["FEMALE"]].astype(float), adjustment_columns(data, panel)], axis=1)


def support_problem(data: pd.DataFrame) -> str:
    n_w, n_m = int(data["FEMALE"].eq(1).sum()), int(data["FEMALE"].eq(0).sum())
    if len(data) < MIN_MODEL_N:
        return f"n={len(data)} < {MIN_MODEL_N}"
    if min(n_w, n_m) < MIN_GROUP_N:
        return f"men n={n_m}, women n={n_w}; each needs ≥{MIN_GROUP_N}"
    return ""


class _HC3Result:
    def __init__(self, ols, cov: np.ndarray):
        names = ols.params.index
        self.params = ols.params
        self.rsquared = ols.rsquared
        self._cov = pd.DataFrame(cov, index=names, columns=names)
        self.bse = pd.Series(np.sqrt(np.diag(cov)), index=names)
        self.pvalues = pd.Series(2 * sps_norm.sf(np.abs(self.params / self.bse)), index=names)

    def cov_params(self) -> pd.DataFrame:
        return self._cov

    def conf_int(self) -> pd.DataFrame:
        z = sps_norm.ppf(0.975)
        return pd.DataFrame({0: self.params - z * self.bse, 1: self.params + z * self.bse})

    def wald_test(self, restriction: np.ndarray, scalar: bool = True):
        value = restriction @ self.params.values
        statistic = float(value @ np.linalg.solve(restriction @ self._cov.values @ restriction.T, value))
        return SimpleNamespace(statistic=statistic, pvalue=float(sps_chi2.sf(statistic, restriction.shape[0])))


def fit_hc3(y: pd.Series, exog: pd.DataFrame):
    ols = sm.OLS(y, exog).fit()
    X = ols.model.wexog
    h = np.einsum("ij,ik,kj->i", X, X, ols.normalized_cov_params)
    singular = (1 - h) < LEVERAGE_ONE_TOL
    if not singular.any():
        return sm.OLS(y, exog).fit(cov_type="HC3"), 0
    resid = np.asarray(ols.wresid, dtype=float)
    scale = np.zeros_like(h)
    scale[~singular] = (resid[~singular] / (1 - h[~singular])) ** 2
    pinv = ols.model.pinv_wexog
    return _HC3Result(ols, pinv @ (scale[:, None] * pinv.T)), int(singular.sum())


def fit_ancova(data: pd.DataFrame, outcome: str, panel: str) -> dict:
    exog = sm.add_constant(sex_design(data, panel), has_constant="add")
    result, n_leverage_one = fit_hc3(data[outcome].astype(float), exog)
    params, cov = result.params, result.cov_params().values
    names = list(params.index)
    contrast_w = np.zeros(len(params))
    contrast_w[names.index("const")] = 1.0
    contrast_w[names.index("FEMALE")] = 1.0
    ci = result.conf_int().loc["FEMALE"]
    return {
        "adj_mean_men": float(params["const"]),
        "adj_mean_men_se": float(result.bse["const"]),
        "adj_mean_women": float(contrast_w @ params),
        "adj_mean_women_se": float(np.sqrt(contrast_w @ cov @ contrast_w)),
        "diff_women_minus_men": float(params["FEMALE"]),
        "diff_se": float(result.bse["FEMALE"]),
        "diff_ci_lo": float(ci.iloc[0]),
        "diff_ci_hi": float(ci.iloc[1]),
        "diff_p": float(result.pvalues["FEMALE"]),
        "n_leverage_one": n_leverage_one,
    }


def fit_check_model(data: pd.DataFrame, outcome: str, panel: str, kind: str) -> dict:
    empty = {"check_model": kind, "check_OR": np.nan, "check_OR_ci_lo": np.nan,
             "check_OR_ci_hi": np.nan, "check_p": np.nan, "check_status": ""}
    design = sex_design(data, panel)
    y = data[outcome].astype(int)
    if kind == "logistic":
        table = pd.crosstab(data["FEMALE"], y)
        if table.shape[1] < 2 or (table == 0).any().any():
            empty["check_status"] = "not estimable (a sex has 0 cases in one outcome level)"
            return empty
        model, kwargs = sm.Logit(y, sm.add_constant(design, has_constant="add")), {"disp": False, "maxiter": 200}
    else:
        if y.nunique() < MIN_OUTCOME_LEVELS:
            empty["check_status"] = f"only {y.nunique()} extent levels"
            return empty
        levels = sorted(y.unique())
        model = OrderedModel(pd.Categorical(y, categories=levels, ordered=True), design, distr="logit")
        kwargs = {"method": "bfgs", "maxiter": 500, "disp": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(**kwargs)
        if not result.mle_retvals.get("converged", False):
            empty["check_status"] = "did not converge"
            return empty
        ci = result.conf_int().loc["FEMALE"]
        return {"check_model": kind, "check_OR": float(np.exp(result.params["FEMALE"])),
                "check_OR_ci_lo": float(np.exp(ci.iloc[0])), "check_OR_ci_hi": float(np.exp(ci.iloc[1])),
                "check_p": float(result.pvalues["FEMALE"]), "check_status": "ok"}
    except Exception as error:
        empty["check_status"] = f"{type(error).__name__}: {error}"
        return empty


def fit_interaction(data: pd.DataFrame, outcome: str, panel: str) -> dict:
    others = [a for a in ANALYSIS_APOE if a != INTERACTION_REFERENCE_APOE]
    present = [a for a in others
               if data["apoe_key"].eq(a).sum() >= MIN_GROUP_N
               and data.loc[data["apoe_key"].eq(a), "FEMALE"].nunique() == 2]
    data = data.loc[data["apoe_key"].isin([INTERACTION_REFERENCE_APOE, *present])]
    if not present or len(data) < MIN_MODEL_N:
        return {"interaction_status": "not fitted: too few haplotypes with both sexes"}

    design = pd.DataFrame({"FEMALE": data["FEMALE"].astype(float)}, index=data.index)
    terms = []
    for apoe in present:
        dummy = data["apoe_key"].eq(apoe).astype(float)
        design[f"APOE_{apoe}"] = dummy
        design[f"FEMALE_x_APOE_{apoe}"] = dummy * design["FEMALE"]
        terms.append(f"FEMALE_x_APOE_{apoe}")
    design = pd.concat([design, adjustment_columns(data, panel)], axis=1)
    exog = sm.add_constant(design, has_constant="add")
    result, n_leverage_one = fit_hc3(data[outcome].astype(float), exog)

    names = list(result.params.index)
    restriction = np.zeros((len(terms), len(names)))
    for row, term in enumerate(terms):
        restriction[row, names.index(term)] = 1.0
    wald = result.wald_test(restriction, scalar=True)
    record = {"interaction_status": "ok", "haplotypes_in_test": ", ".join(APOE_SHORT[a] for a in present),
              "interaction_df": len(terms), "interaction_chi2": float(wald.statistic),
              "interaction_p": float(wald.pvalue), "n_leverage_one": n_leverage_one}
    for apoe in present:
        record[f"interaction_coef_{APOE_SHORT[apoe]}_vs_3/3"] = float(result.params[f"FEMALE_x_APOE_{apoe}"])
        record[f"interaction_p_{APOE_SHORT[apoe]}_vs_3/3"] = float(result.pvalues[f"FEMALE_x_APOE_{apoe}"])
    return record


def outcome_list(columns: dict) -> list[tuple[str, str, str]]:
    return ([("positivity", region, col) for region, col in columns["pos"].items()]
            + [("spatial extent", "Braak 1-2 + 3-4 + 5-6", "spatial")]
            + [("SUVR", region, col) for region, col in columns["suvr"].items()])


def run_all(panels_by_tracer: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results, interactions, counts = [], [], []
    covariates = "age, CL" + (", cohort" if ADJUST_FOR_COHORT else "")

    for tracer, panels in panels_by_tracer.items():
        columns = TRACERS[tracer]
        for panel in PANEL_ORDER:
            panel_data = panels[panel]
            for apoe in ANALYSIS_APOE:
                for sex_value, sex_label in [(0.0, "Men"), (1.0, "Women")]:
                    g = panel_data.loc[panel_data["apoe_key"].eq(apoe) & panel_data["FEMALE"].eq(sex_value)]
                    counts.append({"tracer": tracer, "panel": panel, "APOE": f"APOE{APOE_SHORT[apoe]}",
                                   "sex": sex_label, "n": len(g), "age_mean": g["AGE"].mean(),
                                   "age_sd": g["AGE"].std(), "CL_median": g["CL"].median(),
                                   "spatial_extent_mean": g["spatial"].mean()})

            for outcome_type, region, column in outcome_list(columns):
                data_all = panel_data.dropna(subset=[column])
                interactions.append({"tracer": tracer, "cl_sample": SAMPLE_NAME, "panel": panel,
                                     "outcome_type": outcome_type, "region": region,
                                     "covariates": covariates + (", diagnosis" if panel == OVERALL_LABEL else ""),
                                     **fit_interaction(data_all, column, panel)})
                for apoe in ANALYSIS_APOE:
                    data = data_all.loc[data_all["apoe_key"].eq(apoe)].copy()
                    row = {"tracer": tracer, "cl_sample": SAMPLE_NAME, "APOE": f"APOE{APOE_SHORT[apoe]}",
                           "apoe_key": apoe, "panel": panel, "outcome_type": outcome_type, "region": region,
                           "n_men": int(data["FEMALE"].eq(0).sum()), "n_women": int(data["FEMALE"].eq(1).sum()),
                           "covariates": covariates + (", diagnosis" if panel == OVERALL_LABEL and ADJUST_OVERALL_MODEL_FOR_DX else "")}
                    problem = support_problem(data)
                    if problem:
                        results.append({**row, "status": f"not fitted: {problem}"})
                        continue
                    try:
                        row.update(fit_ancova(data, column, panel))
                        row["status"] = "ok"
                    except Exception as error:
                        results.append({**row, "status": f"ANCOVA failed: {type(error).__name__}: {error}"})
                        continue
                    if outcome_type == "positivity":
                        row.update(fit_check_model(data, column, panel, "logistic"))
                    elif outcome_type == "spatial extent":
                        row.update(fit_check_model(data, column, panel, "ordinal"))
                    results.append(row)

    results_df, interactions_df = pd.DataFrame(results), pd.DataFrame(interactions)
    for frame, p_col, q_col in [(results_df, "diff_p", "diff_q_FDR"), (interactions_df, "interaction_p", "interaction_q_FDR")]:
        frame[q_col] = np.nan
        ok = frame[p_col].notna() if p_col in frame else pd.Series(False, index=frame.index)
        if ok.any():
            frame.loc[ok, q_col] = multipletests(frame.loc[ok, p_col], method="fdr_bh")[1]

    is_pos = results_df["outcome_type"].eq("positivity")
    for column in ["adj_mean_men", "adj_mean_women", "adj_mean_men_se", "adj_mean_women_se",
                   "diff_women_minus_men", "diff_se", "diff_ci_lo", "diff_ci_hi"]:
        results_df.loc[is_pos, column] = results_df.loc[is_pos, column] * 100
    results_df["units"] = np.select([is_pos, results_df["outcome_type"].eq("spatial extent")],
                                    ["percentage points", "number of positive regions (0-3)"], default="SUVR")
    return results_df, interactions_df, pd.DataFrame(counts)


def adjusted_suvr_points(panels_by_tracer: dict) -> pd.DataFrame:
    records = []
    for tracer, panels in panels_by_tracer.items():
        for panel in PANEL_ORDER:
            for apoe in ANALYSIS_APOE:
                for region, column in TRACERS[tracer]["suvr"].items():
                    data = panels[panel].loc[panels[panel]["apoe_key"].eq(apoe)].dropna(subset=[column]).copy()
                    if support_problem(data):
                        continue
                    exog = sm.add_constant(sex_design(data, panel), has_constant="add")
                    fit = sm.OLS(data[column].astype(float), exog).fit()
                    covariates = [c for c in exog.columns if c not in ("const", "FEMALE")]
                    adjusted = data[column].astype(float) - exog[covariates] @ fit.params[covariates]
                    for index in data.index:
                        records.append({"tracer": tracer, "panel": panel, "apoe_key": apoe,
                                        "group": "women" if data.at[index, "FEMALE"] == 1 else "men",
                                        "BIDS_SUBJECT": data.at[index, "BIDS_SUBJECT"],
                                        "BIDS_SESSION": data.at[index, "BIDS_SESSION"],
                                        "COHORT": data.at[index, COHORT_COL], "region": region,
                                        "suvr_raw": float(data.at[index, column]),
                                        "suvr_adjusted": float(adjusted.at[index])})
    return pd.DataFrame(records)


def figure_table(results: pd.DataFrame) -> pd.DataFrame:
    d = results.loc[results["outcome_type"].eq("SUVR")].copy()
    for group in ["men", "women"]:
        d[f"adj_mean_{group}_ci_lo"] = d[f"adj_mean_{group}"] - 1.96 * d[f"adj_mean_{group}_se"]
        d[f"adj_mean_{group}_ci_hi"] = d[f"adj_mean_{group}"] + 1.96 * d[f"adj_mean_{group}_se"]
    d["p_on_figure"] = d["diff_p"].map(format_p)
    order = {"tracer": list(TRACERS), "apoe_key": ANALYSIS_APOE, "panel": PANEL_ORDER, "region": REGIONS}
    for column, levels in order.items():
        d[f"_{column}"] = d[column].map({v: i for i, v in enumerate(levels)})
    d = d.sort_values([f"_{c}" for c in order])
    cols = ["tracer", "APOE", "panel", "region", "n_men", "n_women",
            "adj_mean_men", "adj_mean_men_ci_lo", "adj_mean_men_ci_hi",
            "adj_mean_women", "adj_mean_women_ci_lo", "adj_mean_women_ci_hi",
            "diff_women_minus_men", "diff_ci_lo", "diff_ci_hi", "diff_p", "diff_q_FDR", "p_on_figure", "status"]
    return d[cols].reset_index(drop=True)


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


def draw_boxes(ax, points, stats, ylim, title, rng):
    span = ylim[1] - ylim[0]
    drawn = False
    for x, region in enumerate(REGIONS):
        region_points = points.loc[points["region"].eq(region)]
        if region_points.empty:
            continue
        tops = []
        for offset, group in zip([-OFFSET, OFFSET], ["men", "women"]):
            values = region_points.loc[region_points["group"].eq(group), "suvr_adjusted"].dropna().to_numpy()
            if values.size == 0:
                continue
            color = PALETTE_SEX[group]
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


def sex_figure(points: pd.DataFrame, results: pd.DataFrame, tracer: str, path_out: Path, rng) -> None:
    points = points.loc[points["tracer"].eq(tracer) & points["apoe_key"].isin(ANALYSIS_APOE)]
    stats = results.loc[results["outcome_type"].eq("SUVR") & results["tracer"].eq(tracer)]

    n_rows = len(ANALYSIS_APOE)
    fig_h = TOP_IN + n_rows * ROW_H_IN + (n_rows - 1) * ROW_GAP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300, facecolor="white")
    total_w = len(PANEL_ORDER) * COL_W + (len(PANEL_ORDER) - 1) * COL_GAP
    start_x = (1.0 - total_w) / 2
    row_h = ROW_H_IN / fig_h
    bottoms = [(BOTTOM_IN + (n_rows - 1 - i) * (ROW_H_IN + ROW_GAP_IN)) / fig_h for i in range(n_rows)]
    label_x = max(0.005, start_x - 0.03)
    column_axes = {panel: fixed_axis(tracer, panel, points.loc[points["panel"].eq(panel), "suvr_adjusted"])
                   for panel in PANEL_ORDER}

    for i, apoe in enumerate(ANALYSIS_APOE):
        for j, panel in enumerate(PANEL_ORDER):
            ax = fig.add_axes([start_x + j * (COL_W + COL_GAP), bottoms[i], COL_W, row_h])
            p = points.loc[points["apoe_key"].eq(apoe) & points["panel"].eq(panel)]
            s = stats.loc[stats["apoe_key"].eq(apoe) & stats["panel"].eq(panel)]
            draw_boxes(ax, p, s, column_axes[panel][0], panel, rng)
            apply_ticks(ax, column_axes[panel][1])
            ax.set_ylabel("Adjusted SUVR" if j == 0 else "", fontsize=FONT_SIZE["axis_label"], labelpad=6)
        y = bottoms[i] + row_h + 0.55 / fig_h
        fig.text(label_x, y, "abc"[i] + ".", ha="left", va="bottom", fontsize=FONT_SIZE["title"] + 1, fontweight="bold")
        fig.text(label_x + 0.0225, y, rf"$\it{{APOE}}${APOE_SHORT[apoe]}", ha="left", va="bottom",
                 fontsize=FONT_SIZE["title"] + 1)

    width = 0.30
    lax = fig.add_axes([start_x + total_w / 2 - width / 2, bottoms[-1] - 1.05 / fig_h, width, 0.025])
    lax.axis("off")
    handles = [Patch(facecolor=matplotlib.colors.to_rgba(PALETTE_SEX[g], 0.22), edgecolor=PALETTE_SEX[g],
                     linewidth=1.8, label=label) for label, g in [("Men", "men"), ("Women", "women")]]
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
    check_inputs(base, ptau).to_csv(out / f"{PREFIX}_input_check.csv", index=False)

    panels_by_tracer = {}
    for tracer in TRACERS:
        eligible = prepare_tracer(base, tracer, BASE_FILE.name)
        panels_by_tracer[tracer] = build_panels(eligible, tracer)

    print("\nFitting models...")
    results, interactions, counts = run_all(panels_by_tracer)
    points = adjusted_suvr_points(panels_by_tracer)
    table = figure_table(results)
    for frame, name in [(results, "results"), (interactions, "interaction"), (counts, "counts"),
                        (points, "adjusted_suvr"), (table, "figure_table")]:
        path = out / f"{PREFIX}_{name}.csv"
        frame.to_csv(path, index=False)
        print(f"Saved: {path} ({len(frame):,} rows)")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
        "axes.titleweight": "regular", "font.size": FONT_SIZE["tick"], "axes.labelsize": FONT_SIZE["axis_label"],
        "xtick.labelsize": FONT_SIZE["tick"], "ytick.labelsize": FONT_SIZE["tick"],
    })
    rng = np.random.default_rng(JITTER_SEED)
    if JITTER_SKIP:
        rng.uniform(-JITTER, JITTER, JITTER_SKIP)
    for tracer in TRACERS:
        sex_figure(points, results, tracer, out / f"FIG_S8_{TRACER_SLUG[tracer]}.png", rng)

    suvr = results.loc[results["outcome_type"].eq("SUVR") & results["status"].eq("ok")]
    inter = interactions.loc[interactions["outcome_type"].eq("SUVR") & interactions["interaction_status"].eq("ok")]
    print(f"\nSUVR, women vs men: p < 0.05 in {int((suvr['diff_p'] < 0.05).sum())} of {len(suvr)} fitted tests "
          f"(women higher in {int(((suvr['diff_p'] < 0.05) & (suvr['diff_women_minus_men'] > 0)).sum())}); "
          f"FDR q < 0.05 in {int((suvr['diff_q_FDR'] < 0.05).sum())}.")
    print(f"SUVR, sex × APOE: lowest p = {inter['interaction_p'].min():.2f}, lowest q = {inter['interaction_q_FDR'].min():.2f}.")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        for tracer in TRACERS:
            grid = table.loc[table["tracer"].eq(tracer)].pivot_table(
                index=["APOE", "panel"], columns="region", values="p_on_figure", aggfunc="first", sort=False)
            print(f"\np-values printed on the {tracer} figure:")
            print(grid.to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
