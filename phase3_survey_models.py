# -*- coding: utf-8 -*-
"""
PHASE 3 — survey-weighted adjusted associations (Python port).

Full model (individual + household + sub-region fixed effects) for:
  - continuous composite ECI  -> adjusted percentage-point differences
  - inadequate coverage (ECI < 0.70) -> adjusted odds ratios
with design-based Wald tests for each categorical predictor.

METHOD NOTE: R uses survey::svyglm (full stratified design, Taylor linearisation).
Python has no exact equivalent, so this uses weighted GLMs (survey weights rescaled
to the sample size) with PSU cluster-robust ('sandwich') standard errors. The point
estimates reproduce svyglm; the SEs are a close, slightly conservative approximation
because stratification is not used in the variance. Associations, not causal effects.

Run:  python phase3_survey_models.py
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import eci_common as ec

REFS = {"age_group4": "15-19", "education4": "No education", "wealth5": "Lowest",
        "residence2": "Urban", "parity3": "1", "marital3": "Married/living together",
        "subregion15": "Kampala"}
PRETTY = {"age_group4": "Age", "education4": "Education", "wealth5": "Wealth",
          "residence2": "Residence", "parity3": "Parity", "marital3": "Marital status",
          "subregion15": "Subregion"}


def _formula(outcome):
    terms = " + ".join([f"C({v}, Treatment('{r}'))" for v, r in REFS.items()])
    return f"{outcome} ~ {terms}"


def _label(term):
    # Turn "C(education4, Treatment('No education'))[T.Secondary]" into "Education: Secondary"
    for v, pretty in PRETTY.items():
        if term.startswith(f"C({v}"):
            lvl = term.split("[T.")[-1].rstrip("]")
            return f"{pretty}: {lvl}"
    return term


def _model_frame(d):
    cols = list(REFS) + ["eci", "eci_inadequate_70", "wt", "psu"]
    m = d.dropna(subset=cols).copy()
    m["wt_s"] = m["wt"] * len(m) / m["wt"].sum()   # rescale weights to n (for honest df)
    return m


def _wald(res):
    wt = res.wald_test_terms(scalar=True)
    tab = wt.table.reset_index().rename(columns={"index": "term"})
    rows = []
    for _, r in tab.iterrows():
        raw = str(r["term"])
        var = next((v for v in REFS if raw.startswith(f"C({v}")), raw)
        stat = r.get("statistic", r.get("chi2", np.nan))
        rows.append([PRETTY.get(var, var), round(float(stat), 2), int(r.get("df_constraint", r.get("df", 0))),
                     ec.pct(r["pvalue"], 4) / 100 if False else round(float(r["pvalue"]), 4)])
    return pd.DataFrame(rows, columns=["covariate", "statistic", "df", "p_value"]).drop_duplicates("covariate")


def main():
    ec.ensure_dirs()
    d = ec.load_analytic()
    m = _model_frame(d)
    print(f"Model sample n = {len(m):,}")

    # ---------- continuous ECI (adjusted pp differences)
    gc = smf.glm(_formula("eci"), data=m, family=sm.families.Gaussian(), freq_weights=m["wt_s"])
    rc = gc.fit(cov_type="cluster", cov_kwds={"groups": m["psu"]})
    ci = rc.conf_int()
    cont = pd.DataFrame({
        "covariate": [_label(t) for t in rc.params.index],
        "adj_pp_difference": (100 * rc.params).round(2).values,
        "ci_low_pp": (100 * ci[0]).round(2).values, "ci_high_pp": (100 * ci[1]).round(2).values,
        "p_value": rc.pvalues.round(4).values})
    cont = cont[cont["covariate"] != "Intercept"]
    cont["Adj. pp difference (95% CI)"] = [
        ec.ci_str(e/100, l/100, h/100, sign=True) for e, l, h in
        zip(cont.adj_pp_difference, cont.ci_low_pp, cont.ci_high_pp)]
    cont_out = cont[["covariate", "Adj. pp difference (95% CI)", "p_value"]]

    # ---------- binary ECI<0.70 (adjusted OR)
    gb = smf.glm(_formula("eci_inadequate_70"), data=m, family=sm.families.Binomial(), freq_weights=m["wt_s"])
    rb = gb.fit(cov_type="cluster", cov_kwds={"groups": m["psu"]})
    cib = rb.conf_int()
    binr = pd.DataFrame({
        "covariate": [_label(t) for t in rb.params.index],
        "odds_ratio": np.exp(rb.params).round(3).values,
        "ci_low": np.exp(cib[0]).round(3).values, "ci_high": np.exp(cib[1]).round(3).values,
        "p_value": rb.pvalues.round(4).values})
    binr = binr[binr["covariate"] != "Intercept"]
    binr["Adjusted OR (95% CI)"] = [f"{o:.2f} ({l:.2f}–{h:.2f})"
                                    for o, l, h in zip(binr.odds_ratio, binr.ci_low, binr.ci_high)]
    bin_out = binr[["covariate", "Adjusted OR (95% CI)", "p_value"]]

    # ---------- Wald tests
    wald_c = _wald(rc); wald_b = _wald(rb)

    refs = pd.DataFrame({"variable": list(REFS), "reference_category": list(REFS.values())})
    out = os.path.join(ec.OUTPUT_DIR, "ECI_Python_Phase3_Survey_Models.xlsx")
    ec.write_xlsx(out, {
        "Adjusted_Continuous": cont_out, "Continuous_Wald": wald_c,
        "Adjusted_Binary70": bin_out, "Binary_Wald": wald_b, "Reference_Categories": refs})
    print("PHASE 3 COMPLETE")
    print("  Education Secondary pp:", cont.loc[cont.covariate == "Education: Secondary", "adj_pp_difference"].values)
    print("  Wealth Highest pp:", cont.loc[cont.covariate == "Wealth: Highest", "adj_pp_difference"].values)
    print("  Subregion Bukedi pp:", cont.loc[cont.covariate == "Subregion: Bukedi", "adj_pp_difference"].values)
    print("  Wealth Highest OR:", binr.loc[binr.covariate == "Wealth: Highest", "odds_ratio"].values)
    print("  Subregion Bukedi OR:", binr.loc[binr.covariate == "Subregion: Bukedi", "odds_ratio"].values)
    print("Workbook:", out)


if __name__ == "__main__":
    main()
