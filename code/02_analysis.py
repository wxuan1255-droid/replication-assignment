"""
==============================================================================
02_analysis.py

Replication of Anderson (2017), "The Benefits of College Athletic Success."
STAGE 2 OF 3: ESTIMATION

What this script does
----------------------
Implements the paper's two estimators and runs each across four samples (all
FBS schools, BCS schools only, non-BCS schools only, and a "weeks 5+" robustness
sample that excludes each team's first four games), reproducing the paper's
Tables 1 and 3-6.

  (A) The Sequential Treatment Effects (STE) model -- paper equation (8):
      a single weighted-least-squares regression of the two-year-differenced
      outcome on current wins, wins two years prior, games played, and year
      fixed effects, weighted by the inverse probability of the realized
      win/loss sequence, clustered by school. This is an EXACT replication:
      the specification is stated in closed form in the paper's own text.

  (B) The propensity-score IV/matching model -- paper equations (3)-(5):
      for each week of the season, stratifies observations into 12
      propensity-score bins, estimates a within-bin reduced-form win effect
      and first-stage "expected remaining wins" effect, and combines these
      (reduced form / (1 + first stage)) across bins and weeks. This is an
      APPROXIMATE reimplementation: the original Stata code implements a
      custom bin-matching estimator whose exact numerical behavior (e.g.
      percentile tie-breaking, bin-level clustering) is not fully specified
      in the paper's prose and had to be inferred by reading the original
      do-file. See replication_report for the year-fixed-effects
      residualization detail that was necessary to match the published
      numbers (Section 5 / "Replication Process").

Input:  ../data/processed/analytic_panel.pkl  (produced by 01_data_cleaning.py)
Output: ../output/tables/table1.csv, table3_all.csv, table4_bcs.csv,
        table5_nonbcs.csv, table6.csv  (raw replicated coefficients; these are
        combined with the published numbers for comparison in
        03_tables_figures.py)

No randomness is used (both estimators are deterministic closed-form/OLS
computations); there is nothing to seed.

Python version: 3.10+ (developed and tested on 3.12)
Required packages: pandas, numpy, statsmodels, scipy  (see ../requirements.txt)
==============================================================================
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sstats

warnings.filterwarnings("ignore")

PROCESSED_DIR = "../data/processed"
OUTPUT_TABLES_DIR = "../output/tables"
os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)

# The paper's ten outcome variables (see report Section 3, "Data")
OUTCOME_VARS = ["alumni_ops_athletics", "alum_non_athl_ops", "alumni_total_giving",
                "vse_alum_giving_rate", "usnews_academic_rep_new", "applicants",
                "acceptance_rate", "firsttime_outofstate", "first_time_instate", "sat_25"]

LABELS = {
    "alumni_ops_athletics": "Alumni Athletic Operating Donations",
    "alum_non_athl_ops": "Alumni Nonathletic Operating Donations",
    "alumni_total_giving": "Total Alumni Donations",
    "vse_alum_giving_rate": "Alumni Giving Rate",
    "usnews_academic_rep_new": "Academic Reputation",
    "applicants": "Applicants",
    "acceptance_rate": "Acceptance Rate",
    "firsttime_outofstate": "First-Time Out-of-State Enrollment",
    "first_time_instate": "First-Time In-State Enrollment",
    "sat_25": "25th Percentile SAT",
}


# ==========================================================================
# ESTIMATOR A: Sequential Treatment Effects (STE) model -- equation (8)
# ==========================================================================

def ste_table(df, bcs_filter=None, weight_col="lag_ipw_weight",
              win_col="lag_seasonwins", win_lag2_col="lag3_seasonwins",
              games_col="lag_seasongames", games_lag2_col="lag3_seasongames",
              outcome_year_gap2_col_prefix="lag2_",
              trim_quantile_col=None, trim_apply_col=None):
    """
    Delta Y_i(t+1) = b0 + b1*W_it + b2*W_i(t-2) + b3*S_it + b4*S_i(t-2)
                     + year FE + e
    weighted by weight_col, trimmed to the bottom 90% of the trim column,
    clustered by school. Returns the coefficient on win_col (b1).
    """
    d = df.copy()
    if bcs_filter is not None:
        d = d[bcs_filter(d)].copy()

    trim_quantile_col = trim_quantile_col or weight_col
    trim_apply_col = trim_apply_col or trim_quantile_col
    trim = d[trim_quantile_col].quantile(0.90)
    d = d[d[trim_apply_col] < trim].copy()
    d = d[d[win_col].notna()].copy()

    rows = []
    for v in OUTCOME_VARS:
        dv = f"{outcome_year_gap2_col_prefix}{v}"
        d["_dY"] = d[v] - d[dv]
        sub = d.dropna(subset=["_dY", win_col, win_lag2_col, games_col, games_lag2_col,
                                weight_col, "year", "school_id"]).copy()
        if len(sub) < 20:
            rows.append({"outcome": v, "coef": np.nan, "se": np.nan, "N": len(sub), "pval": np.nan})
            continue
        sub["year_cat"] = sub["year"].astype(int).astype(str)
        formula = f"_dY ~ {win_col} + {win_lag2_col} + {games_col} + {games_lag2_col} + C(year_cat)"
        try:
            res = smf.wls(formula, data=sub, weights=sub[weight_col]).fit(
                cov_type="cluster", cov_kwds={"groups": sub["school_id"]})
            coef, se, n, pval = res.params[win_col], res.bse[win_col], int(res.nobs), res.pvalues[win_col]
        except Exception:
            coef, se, n, pval = np.nan, np.nan, len(sub), np.nan
        rows.append({"outcome": v, "coef": coef, "se": se, "N": n, "pval": pval})
    out = pd.DataFrame(rows)
    out["label"] = out["outcome"].map(LABELS)
    return out


# ==========================================================================
# ESTIMATOR B: propensity-score IV/matching model -- equations (3)-(5)
# ==========================================================================

def residualize_on_year(d, var):
    """Dependent variable for the IV model: residual of the two-year-
    differenced outcome on year fixed effects (see module docstring)."""
    diff = d[var] - d[f"lag2_{var}"]
    sub = diff.dropna()
    if len(sub) < 30:
        return pd.Series(np.nan, index=d.index)
    year_dum = pd.get_dummies(d.loc[sub.index, "year"].astype(int), prefix="yr", drop_first=False).astype(float)
    res = sm.OLS(sub, year_dum).fit()
    resid = pd.Series(np.nan, index=d.index)
    resid.loc[sub.index] = res.resid
    return resid


def make_bins(pscore, region_mask):
    """12 bins from within-region percentile cut points (10, 18, ..., 90)."""
    vals = pscore[region_mask]
    if len(vals) < 24:
        return pd.Series(np.nan, index=pscore.index)
    cuts = np.percentile(vals, [10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90])
    bins = pd.Series(np.nan, index=pscore.index)
    bins[region_mask & (pscore < cuts[0])] = 1
    for i in range(1, 11):
        bins[region_mask & (pscore >= cuts[i - 1]) & (pscore < cuts[i])] = i + 1
    bins[region_mask & (pscore >= cuts[10])] = 12
    return bins


def week_estimate(d, win_col, pscore_col, dep_var):
    """Reduced-form (or first-stage) win effect for one week, combined
    across 12 propensity-score bins with sample-size weights."""
    win, pscore = d[win_col], d[pscore_col]
    valid = win.notna() & pscore.notna()
    if valid.sum() < 30:
        return None

    min_treated = max(pscore[valid & (win == 1)].min() if (valid & (win == 1)).any() else 0.05, 0.05)
    max_treated = min(pscore[valid & (win == 0)].max() if (valid & (win == 0)).any() else 0.95, 0.95)
    region = valid & (pscore >= min_treated) & (pscore <= max_treated)
    if region.sum() < 30:
        return None

    bins = make_bins(pscore, region)
    rows = []
    for b in range(1, 13):
        m = region & (bins == b) & d[dep_var].notna()
        if m.sum() < 4:
            continue
        sub_win = win[m]
        if min((sub_win == 1).sum(), (sub_win == 0).sum()) < 2:
            continue
        X = sm.add_constant(pd.DataFrame({"win": sub_win, "pscore": pscore[m]}))
        try:
            res = sm.OLS(d.loc[m, dep_var], X).fit()
        except Exception:
            continue
        rows.append({"coef": res.params["win"], "se": res.bse["win"], "n": m.sum()})

    if not rows:
        return None
    rr = pd.DataFrame(rows)
    w = rr["n"] / rr["n"].sum()
    return {"coef": float((rr["coef"] * w).sum()),
            "se": float(np.sqrt((rr["se"] ** 2 * w ** 2).sum())),
            "N": int(rr["n"].sum())}


def iv_table(df, bcs_filter=None, weeks=range(1, 13)):
    """Combine week_estimate()'s reduced-form and first-stage results into
    beta = reduced_form / (1 + first_stage) for each week, then combine the
    weekly betas with sample-size weights."""
    d = df.copy()
    if bcs_filter is not None:
        d = d[bcs_filter(d)].copy()

    rows = []
    for v in OUTCOME_VARS:
        d[f"_riv_{v}"] = residualize_on_year(d, v)
        rv = f"_riv_{v}"
        week_betas, week_ses, week_ns = [], [], []
        for w in weeks:
            win_col, pscore_col, fs_col = f"lag_win_wk{w}", f"lag_pscore_wk{w}", f"lag_exp_wins_wk{w}"
            if win_col not in d.columns or fs_col not in d.columns:
                continue
            rf = week_estimate(d, win_col, pscore_col, rv)
            if rf is None:
                continue
            fs = week_estimate(d, win_col, pscore_col, fs_col)
            if fs is None:
                continue
            fs_denom = 1 + fs["coef"]
            if abs(fs_denom) < 0.05:
                continue
            beta = rf["coef"] / fs_denom
            se = np.sqrt((rf["se"] / fs_denom) ** 2 + (fs["se"] * rf["coef"] / fs_denom ** 2) ** 2)
            week_betas.append(beta)
            week_ses.append(se)
            week_ns.append(rf["N"])

        if not week_betas:
            rows.append({"outcome": v, "coef": np.nan, "se": np.nan, "N": np.nan, "pval": np.nan})
            continue

        week_betas, week_ses, week_ns = np.array(week_betas), np.array(week_ses), np.array(week_ns)
        wts = week_ns / week_ns.sum()
        coef = float((week_betas * wts).sum())
        se = float(np.sqrt((week_ses ** 2 * wts ** 2).sum()))
        pval = 2 * (1 - sstats.norm.cdf(abs(coef / se))) if se > 0 else np.nan
        rows.append({"outcome": v, "coef": coef, "se": se, "N": int(week_ns.max()), "pval": pval})

    out = pd.DataFrame(rows)
    out["label"] = out["outcome"].map(LABELS)
    return out


# ==========================================================================
# Run everything and save raw replicated tables
# ==========================================================================

def build_table1(df):
    """Summary statistics (mean, SD, N) for the key variables, by BCS status.
    Depends only on 01_data_cleaning.py's output -- no estimator involved."""
    sumvars = ["lag_seasonwins", "lag_seasongames", "lag_exp_wins"] + OUTCOME_VARS
    sumlabels = {"lag_seasonwins": "Season Wins", "lag_seasongames": "Season Games",
                 "lag_exp_wins": "Expected Wins", **LABELS}
    rows = []
    for label, cond in [("BCS", df["bcs"] == 1), ("Non-BCS", df["bcs"] == 0)]:
        sub = df[cond]
        for v in sumvars:
            s = sub[v].dropna()
            rows.append({"sample": label, "outcome": v, "variable": sumlabels[v],
                         "mean": s.mean(), "sd": s.std(), "N": int(s.count())})
    return pd.DataFrame(rows)


def main():
    print("Loading analytic panel...")
    df = pd.read_pickle(f"{PROCESSED_DIR}/analytic_panel.pkl")

    print("Table 1: summary statistics...")
    build_table1(df).to_csv(f"{OUTPUT_TABLES_DIR}/table1.csv", index=False)

    specs = [("table3_all", lambda d: d["bcs"] <= 1),
             ("table4_bcs", lambda d: d["bcs"] == 1),
             ("table5_nonbcs", lambda d: d["bcs"] == 0)]
    for name, filt in specs:
        print(f"{name}: running STE and IV models...")
        ste = ste_table(df, bcs_filter=filt)
        iv = iv_table(df, bcs_filter=filt)
        merged = ste.merge(iv, on=["outcome", "label"], suffixes=("_ste", "_iv"))
        merged = merged[["outcome", "label", "coef_iv", "se_iv", "N_iv", "pval_iv",
                          "coef_ste", "se_ste", "N_ste", "pval_ste"]]
        merged.to_csv(f"{OUTPUT_TABLES_DIR}/{name}.csv", index=False)

    print("table6 (weeks 5+ robustness check): running STE and IV models...")
    t6_ste = ste_table(df, bcs_filter=lambda d: d["bcs"] <= 1,
                        weight_col="lag_ipw_weight_5",
                        trim_quantile_col="lag_ipw_weight_5", trim_apply_col="lag_ipw_weight",
                        win_col="lag_seasonwins_5", win_lag2_col="lag3_seasonwins_5",
                        games_col="lag_seasongames_5", games_lag2_col="lag3_seasongames_5")
    t6_iv = iv_table(df, bcs_filter=lambda d: d["bcs"] <= 1, weeks=range(5, 13))
    t6 = t6_ste.merge(t6_iv, on=["outcome", "label"], suffixes=("_ste", "_iv"))
    t6 = t6[["outcome", "label", "coef_iv", "se_iv", "N_iv", "pval_iv", "coef_ste", "se_ste", "N_ste", "pval_ste"]]
    t6.to_csv(f"{OUTPUT_TABLES_DIR}/table6.csv", index=False)

    print(f"\nDone. Raw replicated tables written to {OUTPUT_TABLES_DIR}/")


if __name__ == "__main__":
    main()
