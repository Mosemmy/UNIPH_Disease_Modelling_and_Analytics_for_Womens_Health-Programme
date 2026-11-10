# ECI UDHS 2022 — Python analysis pipeline

A Python port of the R effective-coverage pipeline for the Uganda maternal/newborn
Composite Effective Coverage Index (UDHS 2022). It reproduces the same analysis —
ECI scoring, survey-weighted estimation, adjusted association models, the multilevel
variance decomposition, and the spatial analysis — and has been validated against the
R outputs (see **Validation** below).

## Files
| File | What it does |
|---|---|
| `eci_common.py` | Shared module: config, DHS recode helpers, the 16-item ECI construction, the 15-sub-region derivation, survey design, and the Taylor-linearisation survey estimator. |
| `phase1_eci_scoring.py` | Builds the ECI + analytic dataset; national / domain / threshold / item / sub-region / equity estimates. Saves the analytic dataset the other phases load. |
| `phase2_validation.py` | Completeness, fixed-vs-available agreement, selective-missingness checks, missingness model, design QC. |
| `phase3_survey_models.py` | Survey-weighted adjusted models: continuous ECI (pp differences) and ECI<0.70 (odds ratios) with Wald tests. |
| `phase4_multilevel.py` | Random-intercept models M0–M3: ICC, MOR, PCV (continuous + binary). |
| `phase5_spatial.py` | Sub-region choropleths + areal Global Moran's I / LISA (needs the parish shapefile). |
| `run_all.py` | Runs Phase 1 → 5 in order. |
| `requirements.txt` | Python dependencies. |

## Install & run
```bash
pip install -r requirements.txt

# point at your data (default project root is ./project next to the scripts,
# with the IR file expected at ./project/UDHS 2022 Updated Dataset/UGIR7BFL_100V2.dta)
export ECI_IR_PATH="/path/to/UGIR7BFL_100V2.dta"
export ECI_PROJECT_ROOT="/path/to/ECI Project (data, scripts, outputs)"
export ECI_OUTPUT_ROOT_NAME="ECI_Analysis_Outputs_PYTHON_2026"
export ECI_PARISH_SHP="/path/to/uganda_parishes_2024/uganda_parishes_2024.shp"  # Phase 5 only

python run_all.py            # or run each phase script on its own
```
On Windows (PowerShell): `$env:ECI_IR_PATH="C:\...\UGIR7BFL_100V2.dta"` etc.
Outputs (one `.xlsx` per phase, CSVs, PNG figures, and the analytic dataset as
`Objects/ECI_analytic_dataset_FINAL.parquet`) are written under
`<ECI_PROJECT_ROOT>/<ECI_OUTPUT_ROOT_NAME>/`.

Phase 1 must run first (it writes the analytic parquet the others load). Phases 1–4
need only the IR `.dta`; Phase 5 additionally needs the parish `.shp`.

## Validation (Python vs R)
Reproduced on the same IR file:

| Result | R | Python |
|---|---|---|
| Cohort n / clusters | 5,364 / 683 | 5,364 / 683 |
| Fixed-ECI completeness | 96.5% (5,176) | 96.5% (5,176) |
| National composite ECI | 67.1% (66.0–68.2) | 67.11% (66.03–68.19) |
| ANC / Delivery / PNC | 85.0 / 61.6 / 54.4 | 85.04 / 61.62 / 54.40 |
| ECI < 0.60 / 0.70 / 0.80 | 41.4 / 55.4 / 64.1 | 41.4 / 55.4 / 64.1 |
| Kampala / Bukedi | 79.5 / 51.8 | 79.5 / 51.8 |
| Adj. Education-Secondary (pp) | +6.4 | +6.45 |
| Adj. Wealth-Highest (pp) | +10.7 | +10.68 |
| Adj. Bukedi (pp) | −15.2 | −15.17 |
| Adj. Wealth-Highest OR | 0.43 | 0.433 |
| Adj. Bukedi OR | 4.0 | 4.00 |
| Multilevel ICC (null → full) | 26.2 → 15.7% | 26.3 → 16.2% |
| Multilevel MOR (null → full) | 3.35 → 2.41 | 3.44 → 2.48 (VB, approx) |

## Method notes — where Python differs from R
Python has no single package equal to R's `survey`/`lme4`, so three choices are made,
all documented in the relevant script:

1. **Survey means/proportions** (Phase 1, 2, 5): a direct Taylor-linearisation
   estimator in `eci_common.svy_mean` replicates `survey::svymean`/`svyby` on
   `svydesign(ids=~psu, strata=~strata, weights=~wt, nest=TRUE)` with
   `survey.lonely.psu="adjust"`. Means **and** CIs match R to two decimals.
2. **Survey-weighted regression** (Phase 3): weighted GLMs (survey weights rescaled
   to the sample size) with PSU cluster-robust SEs. The **point estimates reproduce
   `svyglm` exactly**; the SEs are a close, slightly conservative approximation
   because stratification is not used in the variance.
3. **Binary multilevel model** (Phase 4): `statsmodels` has no frequentist Laplace/AGQ
   logistic GLMM, so a variational-Bayes logistic mixed model is used and the latent
   ICC/MOR are derived from its cluster variance. These **binary** figures are
   approximate (within ~3% of `lme4` here); the **continuous** MixedLM matches `lme4`
   closely. For publication-grade binary GLMM, use the R Phase 4 script.

## Spatial scope
UDHS 2022 has no released DHS GPS cluster layer, so cluster-level hotspot statistics
(Local Moran at cluster level, Getis-Ord Gi*, SaTScan) and household-to-facility
distance are not possible. Phase 5 therefore does the areal (15-region) analysis:
survey-weighted choropleths and a confirmatory Global Moran's I / LISA. With 15 areal
units these tests are low-power; the primary evidence of geographic clustering is the
Phase 4 multilevel ICC/MOR.

## Data
No data is included in this repository. The UDHS 2022 individual recode (IR) file
is available on request from the [DHS Program](https://dhsprogram.com/data/) and
may not be redistributed. Place your files locally and point the scripts at them
with the environment variables above.
