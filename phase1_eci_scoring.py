# -*- coding: utf-8 -*-
"""
PHASE 1 — ECI scoring + survey-weighted descriptives (Python port).

Builds the 16-item ECI and analytic dataset from the UDHS 2022 IR file, saves it
as Objects/ECI_analytic_dataset_FINAL.parquet (loaded by Phases 2-5), and writes
national, domain, threshold, item, sub-region and equity estimates.

Run:  python phase1_eci_scoring.py
"""
import os
import pandas as pd
import eci_common as ec

KEEP = (["wt", "psu", "strata", "age_years", "parity", "anc_visits", "anc_contact",
         "subregion15", "subregion15_code", "age_group4", "education4", "wealth5",
         "residence2", "parity3", "marital3"]
        + ec.FINAL_ITEMS
        + ["anc_score_fixed", "delivery_score_fixed", "pnc_score_fixed",
           "anc_score_available", "delivery_score_available", "pnc_score_available",
           "eci", "eci_available"]
        + [f"eci_inadequate_{int(t * 100)}" for t in ec.ECI_THRESHOLDS])

EQUITY = {"age_group4": "Maternal age", "education4": "Education", "wealth5": "Wealth",
          "residence2": "Residence", "parity3": "Parity", "marital3": "Marital status"}


def _fmt(df, label_col=None, label=None):
    df = df.copy()
    df["Estimate % (95% CI)"] = [ec.ci_str(e, l, h) for e, l, h in
                                 zip(df.estimate, df.ci_low, df.ci_high)]
    if label_col:
        df.insert(0, label_col, label)
    return df


def main():
    ec.ensure_dirs()
    d, flow = ec.build_analytic()
    a = d[[c for c in KEEP if c in d.columns]].copy()
    for c in ["age_group4", "education4", "wealth5", "residence2", "parity3", "marital3"]:
        a[c] = a[c].astype("string")
    a.to_parquet(ec.ANALYTIC_PARQUET, index=False)
    print(f"Analytic dataset saved: {ec.ANALYTIC_PARQUET} (n = {len(a):,})")

    # National composite, domains and thresholds
    nat = []
    for v, lab in [("eci", "Composite ECI"), ("anc_score_fixed", "ANC domain"),
                   ("delivery_score_fixed", "Delivery/newborn domain"),
                   ("pnc_score_fixed", "PNC domain")] + \
                  [(f"eci_inadequate_{int(t*100)}", f"ECI < {t:.2f}") for t in ec.ECI_THRESHOLDS]:
        nat.append(_fmt(ec.svy_mean(a, v), "indicator", lab))
    national = pd.concat(nat, ignore_index=True)

    # Items
    items = pd.concat([_fmt(ec.svy_mean(a, it), "item", ec.ITEM_LABELS[it])
                       .assign(domain=ec.ITEM_DOMAIN[it]) for it in ec.FINAL_ITEMS],
                      ignore_index=True)

    # Sub-region
    sub = _fmt(ec.svy_mean(a, "eci", domain="subregion15"))
    sub["subregion15"] = pd.Categorical(sub["subregion15"], ec.SUBREGION_LEVELS, ordered=True)
    sub = sub.sort_values("subregion15")

    # Equity
    eq = []
    for v, lab in EQUITY.items():
        r = _fmt(ec.svy_mean(a, "eci", domain=v)).rename(columns={v: "category"})
        r.insert(0, "dimension", lab)
        eq.append(r)
    equity = pd.concat(eq, ignore_index=True)

    out = os.path.join(ec.OUTPUT_DIR, "ECI_Python_Phase1_Scoring.xlsx")
    ec.write_xlsx(out, {"Sample_Flow": flow, "National": national, "Items": items,
                        "Subregion_ECI": sub, "Equity_ECI": equity})
    print("PHASE 1 COMPLETE")
    print(national[["indicator", "Estimate % (95% CI)"]].to_string(index=False))
    print("Workbook:", out)


if __name__ == "__main__":
    main()
