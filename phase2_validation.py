# -*- coding: utf-8 -*-
"""
PHASE 2 — validation and robustness (Python port).

Completeness, fixed- vs available-item agreement, selective-missingness checks
by subgroup, a missingness logistic model, and survey-design QC.

Run:  python phase2_validation.py   (requires phase1 output parquet)
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import eci_common as ec

TABLES = os.path.join(ec.OUTPUT_DIR, "Tables")


def main():
    ec.ensure_dirs()
    d = ec.load_analytic()
    n_elig = len(d)

    # ---- Completeness
    fixed_n = int(d["eci"].notna().sum())
    avail_n = int(d["eci_available"].notna().sum())
    completion = pd.DataFrame([{
        "eligible_n": n_elig, "fixed_eci_n": fixed_n, "fixed_eci_missing_n": n_elig - fixed_n,
        "fixed_eci_completeness_pct": round(100 * fixed_n / n_elig, 2),
        "available_eci_n": avail_n,
        "available_eci_completeness_pct": round(100 * avail_n / n_elig, 2)}])

    # ---- Fixed vs available (survey-weighted) + paired agreement
    fx = ec.svy_mean(d, "eci").iloc[0]
    av = ec.svy_mean(d, "eci_available").iloc[0]
    both = d[d["eci"].notna() & d["eci_available"].notna()]
    diff = (both["eci_available"] - both["eci"])
    agree70 = ((both["eci"] < 0.70) == (both["eci_available"] < 0.70)).mean()
    fixed_avail = pd.DataFrame([
        ["Fixed-item ECI", fixed_n, round(100 * fx.estimate, 3)],
        ["Available-item ECI", avail_n, round(100 * av.estimate, 3)],
        ["Mean paired difference (available - fixed), pp", len(both), round(100 * diff.mean(), 4)],
        ["Max abs paired difference, pp", len(both), round(100 * diff.abs().max(), 4)],
        ["Classification agreement at 0.70, %", len(both), round(100 * agree70, 2)],
    ], columns=["description", "n", "value"])

    # ---- Selective missingness by subgroup
    d = d.copy()
    d["excluded"] = d["eci"].isna().astype(int)
    gap_rows, grp_rows = [], []
    for dim, label in [("age_group4", "Maternal age"), ("education4", "Education"),
                       ("wealth5", "Wealth"), ("residence2", "Residence"),
                       ("parity3", "Parity"), ("subregion15", "Sub-region")]:
        g = d.groupby(dim, observed=True)["excluded"].mean() * 100
        gap_rows.append([label, round(g.min(), 2), round(g.max(), 2), round(g.max() - g.min(), 2)])
        for cat, val in g.items():
            grp_rows.append([label, str(cat), round(val, 2)])
    gaps = pd.DataFrame(gap_rows, columns=["dimension", "min_missing_pct", "max_missing_pct", "gap_pp"])
    by_group = pd.DataFrame(grp_rows, columns=["dimension", "category", "excluded_pct"])

    # ---- Missingness logistic model (unweighted, like R)
    md = d.dropna(subset=["residence2", "wealth5", "education4"]).copy()
    mod = smf.logit(
        "excluded ~ C(residence2, Treatment('Urban')) + C(wealth5, Treatment('Lowest')) "
        "+ C(education4, Treatment('No education'))", data=md).fit(disp=False)
    miss_model = pd.DataFrame({
        "term": mod.params.index, "odds_ratio": np.exp(mod.params).round(3),
        "ci_low": np.exp(mod.conf_int()[0]).round(3), "ci_high": np.exp(mod.conf_int()[1]).round(3),
        "p_value": mod.pvalues.round(4)}).reset_index(drop=True)

    # ---- Survey-design QC
    design = pd.DataFrame([
        ["Unique v022 strata", int(d["strata"].nunique())],
        ["Unique PSUs", int(d["psu"].nunique())],
        ["Strata with <2 PSUs", int((d.groupby("strata")["psu"].nunique() < 2).sum())],
        ["Minimum PSUs per stratum", int(d.groupby("strata")["psu"].nunique().min())],
        ["Sub-regions (should be 15)", int(d["subregion15"].nunique())],
    ], columns=["metric", "value"])

    verdict = "CONDITIONAL GO" if gaps["gap_pp"].max() > 5 else "GO"
    summary = pd.DataFrame([
        ["Fixed-ECI completeness %", completion["fixed_eci_completeness_pct"].iloc[0]],
        ["Fixed vs available: classification agreement %", round(100 * agree70, 2)],
        ["Max subgroup exclusion gap (pp)", gaps["gap_pp"].max()],
        ["Overall verdict", verdict],
    ], columns=["check", "value"])

    out = os.path.join(ec.OUTPUT_DIR, "ECI_Python_Phase2_Validation.xlsx")
    ec.write_xlsx(out, {"Validation_Summary": summary, "Completion": completion,
                        "Fixed_vs_Available": fixed_avail, "Missingness_Gaps": gaps,
                        "Missingness_by_Group": by_group, "Missingness_Model": miss_model,
                        "Survey_Design_QC": design})
    print("PHASE 2 COMPLETE —", verdict)
    print(f"Completeness {completion['fixed_eci_completeness_pct'].iloc[0]}% | "
          f"classification agreement {100*agree70:.1f}% | max gap {gaps['gap_pp'].max()} pp")
    print("Workbook:", out)


if __name__ == "__main__":
    main()
