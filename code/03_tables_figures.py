"""
==============================================================================
03_tables_figures.py

Replication of Anderson (2017), "The Benefits of College Athletic Success."
STAGE 3 OF 3: COMPARISON TO PUBLISHED RESULTS, FINAL TABLES, AND FIGURES

What this script does
----------------------
Takes the raw replicated coefficients produced by 02_analysis.py and:
  1. Compares each of the 40 replicated coefficients (Tables 3-6 x 10
     outcomes) against the published values in published_numbers.py,
     computing correlation, sign agreement, and sample-size match statistics
     (this is the "Replication Accuracy" section of the report).
  2. Produces the paper's "three-win scenario" economic-magnitude calculation
     using the replicated coefficients.
  3. Produces two figures:
       fig_parity.png     -- replicated vs. published coefficients (all 40)
       fig_binscatter.png -- binned relationship between wins and 6 outcomes
                              (a simplified analogue of the paper's Figure 1)

Input:  ../output/tables/table1.csv, table3_all.csv, table4_bcs.csv,
        table5_nonbcs.csv, table6.csv          (from 02_analysis.py)
        ../data/processed/analytic_panel.pkl   (for the binned-scatter figure)
Output: ../output/tables/comparison_all.csv
        ../output/figures/fig_parity.png
        ../output/figures/fig_binscatter.png
        (printed to console: the accuracy summary reported in the report)

Python version: 3.10+ (developed and tested on 3.12)
Required packages: pandas, numpy, statsmodels, matplotlib  (see ../requirements.txt)
==============================================================================
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from published_numbers import TABLE3_ORIG, TABLE4_ORIG, TABLE5_ORIG, TABLE6_ORIG

warnings.filterwarnings("ignore")

OUTPUT_TABLES_DIR = "../output/tables"
OUTPUT_FIGURES_DIR = "../output/figures"
PROCESSED_DIR = "../data/processed"
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)

UNIT_SCALE = {"acceptance_rate": 100}  # acceptance_rate stored in points (0-100); paper reports as a share
DOLLAR_VARS = {"alumni_ops_athletics", "alum_non_athl_ops", "alumni_total_giving"}


# ==========================================================================
# PART 1: compare replicated coefficients to the published values
# ==========================================================================

def compare_table(csv_path, orig_dict, table_name):
    mine = pd.read_csv(csv_path)
    rows = []
    for _, r in mine.iterrows():
        v = r["outcome"]
        if v not in orig_dict:
            continue
        o = orig_dict[v]
        scale = UNIT_SCALE.get(v, 1)
        divisor = 1000 if v in DOLLAR_VARS else 1
        rows.append({
            "table": table_name, "outcome": v,
            "orig_ste": o["ste"], "my_ste": r["coef_ste"] / divisor / scale,
            "orig_ste_n": o["ste_n"], "my_ste_n": r["N_ste"],
            "orig_iv": o["iv"], "my_iv": r["coef_iv"] / divisor / scale,
            "orig_iv_n": o["iv_n"], "my_iv_n": r["N_iv"],
        })
    return pd.DataFrame(rows)


def run_comparison():
    all_cmp = pd.concat([
        compare_table(f"{OUTPUT_TABLES_DIR}/table3_all.csv", TABLE3_ORIG, "Table 3 (All FBS)"),
        compare_table(f"{OUTPUT_TABLES_DIR}/table4_bcs.csv", TABLE4_ORIG, "Table 4 (BCS)"),
        compare_table(f"{OUTPUT_TABLES_DIR}/table5_nonbcs.csv", TABLE5_ORIG, "Table 5 (Non-BCS)"),
        compare_table(f"{OUTPUT_TABLES_DIR}/table6.csv", TABLE6_ORIG, "Table 6 (Weeks 5+)"),
    ], ignore_index=True)

    all_cmp["ste_same_sign"] = np.sign(all_cmp["orig_ste"]) == np.sign(all_cmp["my_ste"])
    all_cmp["iv_same_sign"] = np.sign(all_cmp["orig_iv"]) == np.sign(all_cmp["my_iv"])
    all_cmp["ste_n_diff"] = all_cmp["my_ste_n"] - all_cmp["orig_ste_n"]
    all_cmp["iv_n_diff"] = all_cmp["my_iv_n"] - all_cmp["orig_iv_n"]
    all_cmp.to_csv(f"{OUTPUT_TABLES_DIR}/comparison_all.csv", index=False)

    corr_ste = all_cmp[["orig_ste", "my_ste"]].corr().iloc[0, 1]
    corr_iv = all_cmp[["orig_iv", "my_iv"]].corr().iloc[0, 1]
    print("=" * 70)
    print("REPLICATION ACCURACY SUMMARY (across all 40 outcome x table estimates)")
    print("=" * 70)
    print(f"STE model : correlation = {corr_ste:.3f} | "
          f"sign agreement = {all_cmp['ste_same_sign'].sum()}/{len(all_cmp)} | "
          f"N exact match = {(all_cmp['ste_n_diff'] == 0).sum()}/{len(all_cmp)}")
    print(f"IV model  : correlation = {corr_iv:.3f} | "
          f"sign agreement = {all_cmp['iv_same_sign'].sum()}/{len(all_cmp)} | "
          f"N exact match = {(all_cmp['iv_n_diff'] == 0).sum()}/{len(all_cmp)}")
    return all_cmp


# ==========================================================================
# PART 2: the paper's "three-win scenario" economic-magnitude calculation
# ==========================================================================

def three_win_scenario():
    df = pd.read_pickle(f"{PROCESSED_DIR}/analytic_panel.pkl")
    sub = df[df["bcs"] <= 1]
    base = {
        "alumni_ops_athletics": sub["alumni_ops_athletics"].mean(),
        "applicants": sub["applicants"].mean(),
        "acceptance_rate": sub["acceptance_rate"].mean(),
        "first_time_instate": sub["first_time_instate"].mean(),
        "sat_25": sub["sat_25"].mean(),
    }
    t3 = pd.read_csv(f"{OUTPUT_TABLES_DIR}/table3_all.csv").set_index("outcome")

    print("\n" + "=" * 70)
    print("THREE-WIN SCENARIO (paper's own choice of estimator per outcome:")
    print("IV for donations/applications/acceptance/enrollment, STE for SAT)")
    print("=" * 70)
    results = []
    for v, label in [("alumni_ops_athletics", "Alumni athletic donations ($)"),
                      ("applicants", "Applications (#)"),
                      ("acceptance_rate", "Acceptance rate (pct pts)"),
                      ("first_time_instate", "In-state enrollment (#)"),
                      ("sat_25", "25th pctile SAT (pts)")]:
        model = "coef_ste" if v == "sat_25" else "coef_iv"
        coef = t3.loc[v, model]
        change = coef * 3
        pct = change / base[v] * 100
        print(f"  {label}: {change:,.1f}  ({pct:.1f}% of base {base[v]:,.1f})")
        results.append({"outcome": v, "label": label, "change": change, "pct": pct})
    return pd.DataFrame(results)


# ==========================================================================
# PART 3: figures
# ==========================================================================

def make_parity_figure(cmp_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, xcol, ycol, color, title in [
        (axes[0], "orig_ste", "my_ste", "#2b6cb0", "Sequential Treatment Effects (STE) model"),
        (axes[1], "orig_iv", "my_iv", "#c05621", "Propensity-score IV model"),
    ]:
        lo, hi = min(cmp_df[xcol].min(), cmp_df[ycol].min()), max(cmp_df[xcol].max(), cmp_df[ycol].max())
        pad = (hi - lo) * 0.08
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999999", linestyle="--", linewidth=1)
        ax.scatter(cmp_df[xcol], cmp_df[ycol], s=45, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        corr = cmp_df[[xcol, ycol]].corr().iloc[0, 1]
        ax.set_xlabel(f"Original paper: coefficient")
        ax.set_ylabel(f"Replication: coefficient")
        ax.set_title(f"{title}\ncorrelation = {corr:.3f}", fontsize=12)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIGURES_DIR}/fig_parity.png", dpi=200, bbox_inches="tight")
    plt.close()


def make_binscatter_figure():
    df = pd.read_pickle(f"{PROCESSED_DIR}/analytic_panel.pkl")
    d = df[df["bcs"] <= 1].copy()
    trim = d["lag_ipw_weight"].quantile(0.90)
    d = d[d["lag_ipw_weight"] < trim].copy()
    d = d[d["lag_seasonwins"].notna()].copy()
    d["rseasonwins"] = d["lag_seasonwins"] - d["lag3_seasonwins"]
    d["year_cat"] = d["year"].astype(int).astype(str)

    def residualize(dep, data):
        sub = data.dropna(subset=[dep, "lag3_seasonwins", "lag_seasongames", "lag3_seasongames",
                                   "rseasonwins", "lag_ipw_weight", "year_cat"]).copy()
        formula = f"{dep} ~ lag3_seasonwins + lag_seasongames + lag3_seasongames + C(year_cat)"
        mod = smf.wls(formula, data=sub, weights=sub["lag_ipw_weight"]).fit()
        return sub.index, mod.resid + sub[dep].mean()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [("alumni_ops_athletics", "Athletic Operating\nDonations ($)", 1),
              ("usnews_academic_rep_new", "Academic Reputation\n(points)", 1),
              ("applicants", "Applicants\n(students)", 1),
              ("acceptance_rate", "Acceptance Rate\n(share)", 0.01),
              ("first_time_instate", "In-State Enrollment\n(students)", 1),
              ("sat_25", "25th Pctile SAT\n(points)", 1)]

    for ax, (var, label, scale) in zip(axes.flat, panels):
        d[f"r{var}"] = d[var] - d[f"lag2_{var}"]
        idx_y, resid_y = residualize(f"r{var}", d)
        idx_w, resid_w = residualize("rseasonwins", d.loc[idx_y])
        common = idx_y.intersection(idx_w)
        y, x, wt = resid_y.loc[common] * scale, resid_w.loc[common], d.loc[common, "lag_ipw_weight"]

        bin_id = pd.cut(x, np.arange(-6.5, 7.5, 1))
        tmp = pd.DataFrame({"x": x, "y": y, "w": wt, "bin": bin_id})
        grouped = tmp.groupby("bin", observed=True).apply(
            lambda g: pd.Series({"xm": np.average(g["x"], weights=g["w"]),
                                  "ym": np.average(g["y"], weights=g["w"]), "n": len(g)}))
        grouped = grouped[grouped["n"] >= 5]

        ax.scatter(grouped["xm"], grouped["ym"], s=30, color="#2b6cb0", zorder=3)
        X = np.vstack([x, np.ones(len(x))]).T
        beta, *_ = np.linalg.lstsq(X * np.sqrt(wt.values[:, None]), y.values * np.sqrt(wt.values), rcond=None)
        xx = np.linspace(-6, 6, 50)
        ax.plot(xx, beta[0] * xx + beta[1], color="#c05621", linewidth=1.5, zorder=2)
        ax.axhline(0, color="#cccccc", linewidth=0.7, zorder=1)
        ax.axvline(0, color="#cccccc", linewidth=0.7, zorder=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Change in wins")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Replicated relationship between football wins and outcomes\n"
                  "(binned, weighted by IPW; linear fit overlaid -- cf. Figure 1 in Anderson 2017)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIGURES_DIR}/fig_binscatter.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    cmp_df = run_comparison()
    three_win_scenario()
    print("\nBuilding figures...")
    make_parity_figure(cmp_df)
    make_binscatter_figure()
    print(f"Done. Figures written to {OUTPUT_FIGURES_DIR}/, "
          f"comparison table written to {OUTPUT_TABLES_DIR}/comparison_all.csv")


if __name__ == "__main__":
    main()
