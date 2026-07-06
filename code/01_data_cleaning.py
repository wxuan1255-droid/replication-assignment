"""
==============================================================================
01_data_cleaning.py

Replication of:
  Anderson, Michael L. (2017). "The Benefits of College Athletic Success: An
  Application of the Propensity Score Design." The Review of Economics and
  Statistics, 99(1), 119-134. https://doi.org/10.1162/REST_a_00589

STAGE 1 OF 3: DATA CLEANING AND VARIABLE CONSTRUCTION
==============================================================================

What this script does
----------------------
Rebuilds the paper's team-season analytic panel from the two source Stata
datasets released in the author's public Harvard Dataverse replication package,
following the data-construction logic documented in the original "generate
tables.do" file (obtained from the same replication package). This is a from-
scratch Python re-implementation -- no Stata was used anywhere in this project.

Input (must be placed in ../data/raw/ -- see ../data/README.md):
  - covers_data.dta   : game-by-game results and bookmaker point spreads,
                        1985-2010, source: Covers.com (as compiled by the
                        original author)
  - college data.dta  : school-year donations, admissions, and academic-
                        reputation data, 1986-2009, sources: Voluntary Support
                        of Education (VSE) survey, IPEDS, and US News & World
                        Report (as compiled by the original author)

Output (written to ../data/processed/):
  - analytic_panel.pkl : the final team-season panel used by 02_analysis.py

Unit of observation: one row = one FBS football team's season (a "team-year").
Sample period: football data 1985-2010 (games); merged analytic panel 1986-2009
(the years for which both football and institutional outcome data exist).

No randomness is used anywhere in this script (propensity scores are fit by
deterministic maximum-likelihood logit; there is nothing to seed).

Python version: 3.10+ (developed and tested on 3.12)
Required packages: pandas, numpy, statsmodels  (see ../requirements.txt)
==============================================================================
"""
import datetime
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# Paths (relative -- run this script from inside the code/ folder)
# ------------------------------------------------------------------------
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"

import os
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ==========================================================================
# PART A -- Football data: propensity scores and season aggregates
# (source: covers_data.dta)
# ==========================================================================

def load_covers():
    """Load game-level results/spreads and drop two data-entry errors that the
    original do-file also excludes (a duplicated/erroneous row for two teams
    on a specific date, identified by the original authors)."""
    df = pd.read_stata(f"{RAW_DIR}/covers_data.dta")
    base = pd.Timestamp("1960-01-01")
    df["stata_date"] = (df["date"] - base).dt.days
    df = df[~(((df["team"] == 68) | (df["team"] == 136))
              & (df["stata_date"] == 14568) & (df["line"].isna()))]
    return df.reset_index(drop=True)


def build_week(df):
    """Assign a within-season week number (1, 2, 3, ...) to each game, reset
    for each new team-season, once sorted by (team, season, date)."""
    df = df.sort_values(["team", "season", "date"], kind="mergesort").reset_index(drop=True)
    new_group = (df["team"] != df["team"].shift(1)) | (df["season"] != df["season"].shift(1))
    grp_id = new_group.cumsum()
    df["week"] = df.groupby(grp_id).cumcount() + 1
    return df


def restrict_sample(df):
    """Sample restrictions (matching the original paper, Section II):
      - drop games in week > 12 and games played in Dec/Jan (postseason bowl
        games and conference-championship games, whose participation is
        endogenous to regular-season wins);
      - drop team-seasons with fewer than 8 games that have a recorded
        bookmaker line (too little information to estimate a season-level
        propensity score reliably)."""
    df = df[(df["week"] <= 12) & (df["month"] != 12) & (df["month"] != 1)].copy()
    df = df.sort_values(["teamname", "season", "week"])
    df["total_obs"] = df.groupby(["teamname", "season"])["line"].transform(lambda s: s.notna().sum())
    df = df[df["total_obs"] >= 8].copy()
    df.loc[df["line"].isna(), "win"] = np.nan
    return df


def fit_pscore(df):
    """Propensity score, step 1: logistic regression of the win indicator
    (treatment) on a fifth-order polynomial in the bookmaker's point spread.
    This is the paper's key identification device -- see report Section 4
    (Empirical Strategy)."""
    sub = df.dropna(subset=["win", "line"]).copy()
    for i in range(2, 6):
        sub[f"line{i}"] = sub["line"] ** i
    X = sm.add_constant(sub[["line", "line2", "line3", "line4", "line5"]])
    model = sm.Logit(sub["win"], X).fit(disp=0)
    df.loc[sub.index, "pscore"] = model.predict(X)
    return df, model


def build_clean_line(df):
    """Propensity score, step 2 (the paper's 'cleaned line'): a bookmaker's
    spread in week s partly reflects how much a team has over/under-performed
    its own spread in earlier weeks. To avoid the propensity score partly
    reflecting information revealed *within* the season, the spread in each
    week is regressed on a cubic in the team's cumulative prior-week
    over/under-performance, and the residual (+ constant) is used instead."""
    df["outperform"] = df["realspread"] + df["line"]

    for i in range(1, 12):
        tmp = df["outperform"].where(df["week"] == i)
        avg = tmp.groupby([df["teamname"], df["season"]]).transform("mean")
        df[f"outperform_wk{i}"] = avg.fillna(0)
        df[f"outperformwk{i}_2"] = df[f"outperform_wk{i}"] ** 2
        df[f"outperformwk{i}_3"] = df[f"outperform_wk{i}"] ** 3

    df["line_clean"] = np.where(df["week"] == 1, df["line"], np.nan)
    for i in range(2, 13):
        regs = []
        for lag in range(1, i):
            regs += [f"outperform_wk{lag}", f"outperformwk{lag}_2", f"outperformwk{lag}_3"]
        sub = df[df["week"] == i].dropna(subset=["line"] + regs)
        if len(sub) < len(regs) + 2:
            continue
        X = sm.add_constant(sub[regs])
        res = sm.OLS(sub["line"], X).fit()
        df.loc[sub.index, "line_clean"] = res.resid + res.params["const"]
    return df


def fit_pscore_clean(df):
    sub = df.dropna(subset=["win", "line_clean"]).copy()
    for i in range(2, 6):
        sub[f"line_clean_p{i}"] = sub["line_clean"] ** i
    X = sm.add_constant(sub[["line_clean", "line_clean_p2", "line_clean_p3", "line_clean_p4", "line_clean_p5"]])
    model = sm.Logit(sub["win"], X).fit(disp=0)
    df.loc[sub.index, "pscore_clean_line"] = model.predict(X)
    return df


def build_season_panel(df):
    """Collapse the game-level panel to one row per team-season, carrying
    forward total wins/games, the season-level expected-wins measures, and
    the week-by-week win/propensity-score pairs needed by the IV model in
    02_analysis.py."""
    g = df.groupby(["teamname", "season"])
    season = g.agg(
        seasonwins=("win", "sum"), seasongames=("win", "count"),
        seasonspread=("realspread", "sum"), seasonline=("line", "sum"),
        seasonoutperform=("outperform", "sum"), bcs=("bcs", "mean"),
    ).reset_index()

    wk5 = df[df["week"] >= 5].groupby(["teamname", "season"]).agg(
        seasonwins_5=("win", "sum"), seasongames_5=("win", "count")
    ).reset_index()
    season = season.merge(wk5, on=["teamname", "season"], how="left")
    season["pct_win"] = season["seasonwins"] / season["seasongames"]

    exp = df.groupby(["teamname", "season"]).agg(
        exp_wins_naive=("pscore", "sum"), exp_wins=("pscore_clean_line", "sum")
    ).reset_index()
    season = season.merge(exp, on=["teamname", "season"], how="left")
    season["exp_win_pct"] = season["exp_wins"] / season["seasongames"]

    for w in range(1, 12):
        tmp = df[df["week"] > w].groupby(["teamname", "season"])["pscore"].sum().rename(f"exp_wins_wk{w}")
        season = season.merge(tmp, on=["teamname", "season"], how="left")
    season["exp_wins_wk12"] = 0.0
    tmp0 = df[df["week"] > 0].groupby(["teamname", "season"])["pscore"].sum().rename("exp_wins_wk0")
    season = season.merge(tmp0, on=["teamname", "season"], how="left")

    for i in range(1, 13):
        p = df[df["week"] == i].groupby(["teamname", "season"])["pscore"].mean().rename(f"pscore_wk{i}")
        wgt = df[df["week"] == i].groupby(["teamname", "season"])["win"].mean().rename(f"win_wk{i}")
        pc = df[df["week"] == i].groupby(["teamname", "season"])["pscore_clean_line"].mean().rename(f"pscore_clean_wk{i}")
        season = season.merge(p, on=["teamname", "season"], how="left")
        season = season.merge(wgt, on=["teamname", "season"], how="left")
        season = season.merge(pc, on=["teamname", "season"], how="left")

    return season.rename(columns={"season": "year"})


# ==========================================================================
# PART B -- Merge with institutional outcome data (college data.dta),
# construct lags/leads and inverse-probability weights
# ==========================================================================

# Outcome variables used throughout the replication (see report Section 3,
# "Data": these are the paper's treatment/outcome/control variables).
#   Treatment:   seasonwins (football wins in a season)
#   Controls:    seasongames, lagged wins/games, year fixed effects
#   Outcomes:    the 10 variables below (donations, admissions, academics)
OUTCOME_VARS = ["alumni_ops_athletics", "alum_non_athl_ops", "alumni_total_giving",
                "vse_alum_giving_rate", "usnews_academic_rep_new", "applicants",
                "acceptance_rate", "firsttime_outofstate", "first_time_instate",
                "sat_25", "sat_75"]

COLLEGE_KEEP = ["teamname", "year", "athletics_total", "alumni_ops_athletics", "alumni_ops_total",
                "ops_athletics_total_grand", "usnews_academic_rep_new", "acceptance_rate", "appdate",
                "satdate", "satmt75", "satvr75", "satmt25", "satvr25", "applicants_male",
                "applicants_female", "enrolled_male", "enrolled_female", "vse_alum_giving_rate",
                "first_time_students_total", "first_time_outofstate", "first_time_instate",
                "total_giving_pre_2004", "alumni_total_giving", "asian", "hispanic", "black", "control"]


def load_college():
    df = pd.read_stata(f"{RAW_DIR}/college data.dta", convert_categoricals=False)
    df = df[COLLEGE_KEEP].copy()
    return df.sort_values(["teamname", "year"])


def merge_panels(season, college):
    """Inner join on (school, year): keeps only team-seasons for which both
    football data and institutional outcome data exist."""
    merged = season.merge(college, on=["teamname", "year"], how="inner")
    return merged.sort_values(["teamname", "year"]).reset_index(drop=True)


def add_lags_leads(df, varlist):
    """Construct 1-4 year lags and 1-2 year leads of each variable in
    `varlist`, requiring the intervening years to be contiguous (no gaps) --
    equivalent to the original do-file's `teamname==teamname[_n-k] &
    year==year[_n-1]+1 & ...` chained-contiguity checks. This equivalence
    holds because the merged panel has no duplicate team-year rows and is
    sorted by ascending year within team (verified in the accompanying
    report, Section 5)."""
    df = df.sort_values(["teamname", "year"]).reset_index(drop=True)
    team, year, n = df["teamname"].values, df["year"].values, len(df)

    def shifted(k):
        out = {v: np.full(n, np.nan) for v in varlist}
        idx = np.arange(n)
        valid = (idx - k >= 0) & (idx - k < n)
        ok = valid.copy()
        for i in np.where(valid)[0]:
            j = i - k
            same_team = team[j] == team[i]
            same_gap = (year[i] == year[j] + k) if k > 0 else (year[i] == year[j] - (-k))
            ok[i] = same_team and same_gap
        for v in varlist:
            vals = df[v].values
            out[v][ok] = vals[(idx - k)[ok]]
        return out

    for k, tag in [(1, "lag_"), (2, "lag2_"), (3, "lag3_"), (4, "lag4_"), (-1, "lead_"), (-2, "lead2_")]:
        res = shifted(k)
        for v in varlist:
            df[f"{tag}{v}"] = res[v]
    return df


def resolve_reporting_dates(df):
    """IPEDS reports SAT scores and applicant counts under two different
    fall-reporting-date conventions (see ../data/README.md for the code
    definitions). This reproduces the original do-file's two-pass logic for
    resolving which row a given value belongs to."""
    df = df.sort_values(["teamname", "year"]).reset_index(drop=True)
    team, year, n = df["teamname"].values, df["year"].values, len(df)

    def next_row(i):
        if i + 1 < n and team[i + 1] == team[i] and year[i + 1] == year[i] + 1:
            return i + 1
        return None

    for group, date_col in [(["satmt25", "satmt75", "satvr25", "satvr75"], "satdate"),
                             (["applicants_male", "applicants_female", "enrolled_male", "enrolled_female"], "appdate")]:
        for v in group:
            vals = df[v].values.astype(float)
            temp = vals.copy()
            temp[df[date_col].values == 1] = np.nan
            for i in range(n):
                if np.isnan(temp[i]):
                    j = next_row(i)
                    if j is not None and df[date_col].values[j] == 1:
                        temp[i] = vals[j]
            df[v] = temp
        for v in group:
            vals = df[v].values.astype(float)
            out = np.full(n, np.nan)
            for i in range(n):
                j = next_row(i)
                if j is not None:
                    out[i] = vals[j]
            df[v] = out
    return df


def derive_vars(df):
    """Construct derived outcome variables (non-athletic donations, combined
    SAT scores, total applicants) from the raw college-data fields."""
    df["alum_non_athl_ops"] = df["alumni_ops_total"] - df["alumni_ops_athletics"]
    df["sat_75"] = df["satmt75"] + df["satvr75"]
    df["sat_25"] = df["satmt25"] + df["satvr25"]
    df["applicants"] = df["applicants_male"] + df["applicants_female"]
    return df.rename(columns={"first_time_outofstate": "firsttime_outofstate"})


def recode_bcs(df):
    """Recode BCS-conference status for schools that changed conference
    affiliation during the sample period (paper's footnote 3): Cincinnati,
    Louisville, South Florida, and Connecticut joined a BCS conference mid-
    sample (coded non-BCS here, matching their status at the start of the
    sample); Temple left the Big East in 2004 (coded BCS)."""
    df.loc[df["teamname"].isin(["Cincinnati", "Louisville", "South Florida", "Connecticut"]), "bcs"] = 0
    df.loc[df["teamname"] == "Temple", "bcs"] = 1
    return df


def build_ipw(df, direction_prefix, ngames_col, weeks=range(1, 13)):
    """Inverse-probability-of-treatment weight for the STE model: the
    reciprocal of the probability of the realized win/loss sequence across
    `weeks`, i.e. the product over weeks of (win x pscore + loss x (1-
    pscore)). A missing week (team didn't play, or team-season doesn't
    exist) contributes a factor of 1 (no information)."""
    prod = np.ones(len(df))
    exists = df[ngames_col].notna().values
    for w in weeks:
        win = df[f"{direction_prefix}win_wk{w}"].values
        ps = df[f"{direction_prefix}pscore_wk{w}"].values
        missing = np.isnan(win) | np.isnan(ps)
        term = np.where(missing, 1.0, win * ps + (1 - win) * (1 - ps))
        prod = prod * term
    return np.where(exists, 1.0 / prod, np.nan)


def main():
    print("Step 1/6: loading and cleaning game-level football data...")
    df = load_covers()
    df = build_week(df)
    df = restrict_sample(df)

    print("Step 2/6: fitting propensity scores...")
    df, m1 = fit_pscore(df)
    print(f"  raw-line pscore model: N={int(m1.nobs)}, "
          f"range=[{df['pscore'].min():.4f}, {df['pscore'].max():.4f}] "
          f"(paper reports [0.005, 0.994])")
    df = build_clean_line(df)
    df = fit_pscore_clean(df)

    print("Step 3/6: aggregating to the team-season level...")
    season = build_season_panel(df)

    print("Step 4/6: merging with institutional (donations/admissions) data...")
    college = load_college()
    merged = merge_panels(season, college)
    print(f"  merged panel: {merged.shape[0]} team-year observations")

    print("Step 5/6: constructing lags/leads, reporting-date fixes, derived variables...")
    season_lag_vars = (["seasonwins", "seasongames", "seasonspread", "seasonline", "seasonoutperform",
                         "seasonwins_5", "seasongames_5", "pct_win", "exp_wins", "exp_win_pct", "exp_wins_naive"]
                        + [f"exp_wins_wk{w}" for w in range(1, 13)] + ["exp_wins_wk0"]
                        + [f"pscore_wk{w}" for w in range(1, 13)]
                        + [f"win_wk{w}" for w in range(1, 13)]
                        + [f"pscore_clean_wk{w}" for w in range(1, 13)])
    merged = add_lags_leads(merged, season_lag_vars)
    merged = resolve_reporting_dates(merged)
    merged = derive_vars(merged)
    merged = add_lags_leads(merged, OUTCOME_VARS)
    merged = recode_bcs(merged)
    merged["school_id"] = merged["teamname"].astype("category").cat.codes

    print("Step 6/6: constructing inverse-probability-of-treatment weights...")
    merged["lag_ipw_weight"] = build_ipw(merged, "lag_", "lag_seasongames")
    merged["lead2_ipw_weight"] = build_ipw(merged, "lead2_", "lead2_seasongames")
    merged["lag2_ipw_weight"] = build_ipw(merged, "lag2_", "lag2_seasongames")
    merged["lag_ipw_weight_5"] = build_ipw(merged, "lag_", "lag_seasongames", weeks=range(5, 13))

    out_path = f"{PROCESSED_DIR}/analytic_panel.pkl"
    merged.to_pickle(out_path)
    print(f"\nDone. Saved {out_path}  (shape: {merged.shape})")


if __name__ == "__main__":
    main()
