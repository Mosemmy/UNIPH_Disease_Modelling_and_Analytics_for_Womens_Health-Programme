# -*- coding: utf-8 -*-
"""
PHASE 4 — multilevel variance decomposition (Python port).

Random-intercept models (women nested in clusters = PSUs), M0 null -> M1
individual/household -> M2 contextual -> M3 full, reporting the ICC, median
odds ratio (MOR) and proportional change in variance (PCV).

METHOD NOTE:
  * Continuous ECI: statsmodels MixedLM (REML) reproduces R lme4::lmer closely.
  * Binary ECI<0.70: statsmodels has no frequentist Laplace/AGQ logistic GLMM
    (R uses lme4::glmer). This script fits a variational-Bayes logistic mixed
    model (BinomialBayesMixedGLM) and derives the latent ICC/MOR from its
    cluster variance. These BINARY figures are APPROXIMATE and differ somewhat
    from lme4; for publication-grade binary GLMM use the R Phase 4 script.

Run:  python phase4_multilevel.py
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import eci_common as ec

INDIVIDUAL = ["age_group4", "education4", "wealth5", "parity3", "marital3"]
CONTEXTUAL = ["residence2", "subregion15"]
FULL = INDIVIDUAL + CONTEXTUAL
MODELS = {"M0 Null": [], "M1 Individual/household": INDIVIDUAL,
          "M2 Contextual": CONTEXTUAL, "M3 Full": FULL}
PI2_3 = np.pi ** 2 / 3.0
Z75 = norm.ppf(0.75)


def _rhs(vars_):
    return "1" if not vars_ else " + ".join(f"C({v})" for v in vars_)


def _continuous(data):
    rows = []
    null_var = None
    for name, vs in MODELS.items():
        res = smf.mixedlm(f"eci ~ {_rhs(vs)}", data, groups=data["cluster_id"]).fit(reml=True)
        cv = float(res.cov_re.iloc[0, 0]); rv = float(res.scale)
        icc = 100 * cv / (cv + rv)
        if null_var is None:
            null_var = cv
        pcv = 0.0 if name == "M0 Null" else 100 * (null_var - cv) / null_var
        k = len(res.params) + 2
        aic = -2 * res.llf + 2 * k
        rows.append([name, round(cv, 4), round(rv, 4), round(icc, 1), round(pcv, 1), round(aic, 0)])
    return pd.DataFrame(rows, columns=["model", "cluster_variance", "residual_variance",
                                       "ICC_percent", "PCV_percent", "AIC_approx"])


def _binary(data):
    rows = []
    null_var = None
    for name, vs in MODELS.items():
        try:
            md = BinomialBayesMixedGLM_fit(f"eci_inadequate_70 ~ {_rhs(vs)}", data)
            cv = md  # cluster variance (random-intercept)
        except Exception as e:
            rows.append([name, np.nan, np.nan, np.nan, f"fit failed: {e}"]); continue
        latent_icc = 100 * cv / (cv + PI2_3)
        mor = np.exp(np.sqrt(2 * cv) * Z75)
        if null_var is None:
            null_var = cv
        pcv = 0.0 if name == "M0 Null" else 100 * (null_var - cv) / null_var
        rows.append([name, round(cv, 3), round(latent_icc, 1), round(mor, 2), round(pcv, 1)])
    return pd.DataFrame(rows, columns=["model", "cluster_variance", "latent_ICC_percent",
                                       "MOR_approx", "PCV_percent"])


def BinomialBayesMixedGLM_fit(formula, data):
    """Fit a VB logistic random-intercept model; return the cluster variance."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    vc = {"cluster": "0 + C(cluster_id)"}
    m = BinomialBayesMixedGLM.from_formula(formula, vc, data)
    r = m.fit_vb(verbose=False)
    # vcp_mean[0] is the posterior mean of log-SD of the random intercept
    log_sd = float(r.vcp_mean[0])
    return float(np.exp(2 * log_sd))


def main():
    ec.ensure_dirs()
    d = ec.load_analytic().copy()
    d["cluster_id"] = d["psu"].astype("category")
    cont = d.dropna(subset=["eci"] + FULL).copy()
    binr = d.dropna(subset=["eci_inadequate_70"] + FULL).copy()
    print(f"Continuous N = {len(cont):,} ({cont['cluster_id'].nunique()} clusters) | "
          f"Binary N = {len(binr):,}")

    cont_het = _continuous(cont)
    print("\nContinuous variance partition:")
    print(cont_het.to_string(index=False))

    print("\nFitting binary VB GLMM (approximate) ...")
    bin_het = _binary(binr)
    print(bin_het.to_string(index=False))

    note = pd.DataFrame({"note": [
        "Continuous models: statsmodels MixedLM (REML) ~ R lme4::lmer.",
        "Binary models: variational-Bayes logistic GLMM; latent ICC = var/(var+pi^2/3), "
        "MOR = exp(sqrt(2*var)*0.6745). APPROXIMATE vs lme4::glmer (Laplace ML).",
        "cluster_id = PSU; references are the statsmodels defaults (alphabetical).",
        "AIC_approx = -2*logLik + 2*k; compares models within the continuous family."]})
    out = os.path.join(ec.OUTPUT_DIR, "ECI_Python_Phase4_Multilevel.xlsx")
    ec.write_xlsx(out, {"Continuous_Heterogeneity": cont_het,
                        "Binary_Heterogeneity": bin_het, "Method_Notes": note})
    print("\nPHASE 4 COMPLETE\nWorkbook:", out)


if __name__ == "__main__":
    main()
