# -*- coding: utf-8 -*-
"""
ECI UDHS 2022 — shared module (Python port of the R pipeline).

Holds the configuration, the DHS recode helpers, the 16-item ECI construction,
the 15-sub-region derivation, the survey-design setup, and survey-weighted
estimation helpers used by every phase. Ported faithfully from the R scripts
(Phase 1 Analysis Script.R and siblings).

Design-based estimation uses `samplics` (Taylor linearisation), which matches
R's `survey` package for means/proportions. Where a method has no exact Python
equivalent (survey-weighted regression, binary GLMM) the relevant phase script
documents the approximation used.
"""
import os
import numpy as np
import pandas as pd
import pyreadstat

# ----------------------------------------------------------------------------- CONFIG
PROJECT_ROOT = os.environ.get(
    "ECI_PROJECT_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "project"),
)
OUTPUT_ROOT_NAME = os.environ.get("ECI_OUTPUT_ROOT_NAME", "ECI_Analysis_Outputs_PYTHON_2026")
DATA_DIR = os.path.join(PROJECT_ROOT, "UDHS 2022 Updated Dataset")
# Default IR path mirrors the R default; override with ECI_IR_PATH.
IR_PATH = os.environ.get(
    "ECI_IR_PATH",
    os.path.join(DATA_DIR, "UGIR7BFL_100V2.dta"),
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, OUTPUT_ROOT_NAME)
OBJECTS_DIR = os.path.join(OUTPUT_DIR, "Objects")
ANALYTIC_PARQUET = os.path.join(OBJECTS_DIR, "ECI_analytic_dataset_FINAL.parquet")

# ----------------------------------------------------------------------------- CONSTANTS
REFERENCE_MONTHS = 24
ECI_THRESHOLDS = [0.60, 0.70, 0.80]
MIN_ITEMS_ANC_AVAILABLE = 4
MIN_ITEMS_DELIVERY_AVAILABLE = 3
MIN_ITEMS_PNC_AVAILABLE = 2

FACILITY_CODES = [21, 22, 26, 31, 32, 36]
HOME_CODES = [11, 12]
OTHER_PLACE_CODES = [96]
SKILLED_PNC_PROVIDER_CODES = [11, 12, 13, 14]

SUBREGION_LEVELS = [
    "Kampala", "South Buganda", "North Buganda", "Busoga", "Bukedi",
    "Bugisu", "Teso", "Karamoja", "Lango", "Acholi", "West Nile",
    "Bunyoro", "Tooro", "Ankole", "Kigezi",
]
ANC_ITEMS = ["anc_bp", "anc_urine", "anc_blood", "anc_tetanus", "anc_iron", "anc_iptp", "anc_deworm"]
DELIVERY_ITEMS = ["delivery_sba", "delivery_facility", "nc_cord", "nc_temp", "nc_bf"]
PNC_ITEMS = ["mother_pnc_2d", "newborn_pnc_2d", "pnc_danger_counsel", "pnc_bf_counsel"]
FINAL_ITEMS = ANC_ITEMS + DELIVERY_ITEMS + PNC_ITEMS

ITEM_LABELS = {
    "anc_bp": "Blood pressure measured", "anc_urine": "Urine sample taken",
    "anc_blood": "Blood sample taken", "anc_tetanus": "Protected against neonatal tetanus",
    "anc_iron": "Received/took iron supplements", "anc_iptp": "Took SP/Fansidar during pregnancy",
    "anc_deworm": "Took drugs for intestinal parasites",
    "delivery_sba": "Skilled birth attendant", "delivery_facility": "Delivered in a health facility",
    "nc_cord": "Newborn cord examined", "nc_temp": "Newborn temperature measured",
    "nc_bf": "Breastfeeding observed",
    "mother_pnc_2d": "Maternal postnatal check within 2 days",
    "newborn_pnc_2d": "Newborn postnatal check within 2 days",
    "pnc_danger_counsel": "Counselled on seeking medical attention",
    "pnc_bf_counsel": "Counselled on breastfeeding",
}
ITEM_DOMAIN = {**{k: "ANC" for k in ANC_ITEMS},
               **{k: "Delivery" for k in DELIVERY_ITEMS},
               **{k: "PNC" for k in PNC_ITEMS}}


def ensure_dirs():
    for d in (OUTPUT_DIR, OBJECTS_DIR,
              os.path.join(OUTPUT_DIR, "Tables"), os.path.join(OUTPUT_DIR, "Charts")):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------- RECODE HELPERS
def num(x):
    """Coerce a (possibly labelled/text) Series to numeric, like R num()."""
    return pd.to_numeric(x, errors="coerce")


def yes_no(x):
    x = num(x)
    out = pd.Series(np.nan, index=x.index)
    out[x == 1] = 1.0
    out[x.isin([0, 2])] = 0.0
    # 8/9/98/99/998/999 and anything else stay NaN
    return out


def within_two_days(x):
    x = num(x)
    out = pd.Series(0.0, index=x.index)
    out[((x >= 100) & (x <= 171)) | ((x >= 200) & (x <= 202))] = 1.0
    out[x.isin([998, 999]) | x.isna()] = np.nan
    return out


def is_skilled_pnc_provider(x):
    x = num(x)
    out = pd.Series(0.0, index=x.index)          # TRUE ~ 0 default
    out[x.isin(SKILLED_PNC_PROVIDER_CODES)] = 1.0
    out[x.isin([21, 22, 96])] = 0.0
    out[x.isin([98, 99, 998, 999]) | x.isna()] = np.nan
    return out


def dhs_pnc_2days(check_before, timing_before, provider_before,
                  check_after, timing_after, provider_after):
    """Standard DHS 'PNC check within 2 days' binary indicator (ported from R)."""
    cb, tb, pb = num(check_before).to_numpy(), num(timing_before).to_numpy(), num(provider_before).to_numpy()
    ca, ta, pa = num(check_after).to_numpy(), num(timing_after).to_numpy(), num(provider_after).to_numpy()
    n = len(cb)

    had_check = ((~np.isnan(cb)) & (cb == 1)) | ((~np.isnan(ca)) & (ca == 1))
    pnc_time = np.zeros(n)
    pnc_time[had_check] = 999.0

    idx_before = had_check & (~np.isnan(pb)) & (pb >= 11) & (pb <= 29)
    pnc_time[idx_before] = tb[idx_before]

    idx_before_other = had_check & (~np.isnan(pb)) & (pb > 30) & (pb < 100) & (pnc_time < 1000)
    pnc_time[idx_before_other] = 0.0

    idx_after = (pnc_time == 999) & (~np.isnan(pa)) & (pa >= 11) & (pa <= 29)
    pnc_time[idx_after] = ta[idx_after]

    idx_after_other = (~np.isnan(ta)) & (ta < 1000) & (~np.isnan(pa)) & (pa > 30) & (pa < 100)
    pnc_time[idx_after_other] = 0.0

    pnc_time[~had_check] = 0.0

    within = np.where(
        ((pnc_time >= 100) & (pnc_time <= 171)) | ((pnc_time >= 200) & (pnc_time <= 202)),
        1.0, 0.0)
    return within


def row_score_fixed(df, cols):
    """Mean of items only when ALL are observed (complete cases), else NaN."""
    m = df[cols].apply(num)
    out = m.mean(axis=1)
    out[m.isna().any(axis=1)] = np.nan
    return out


def row_score_available(df, cols, min_obs):
    m = df[cols].apply(num)
    n_obs = m.notna().sum(axis=1)
    score = m.sum(axis=1, skipna=True) / n_obs.clip(lower=1)
    return score.where(n_obs >= min_obs, np.nan)


# ----------------------------------------------------------------------------- BUILD ANALYTIC
def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def build_analytic(ir_path=None):
    """Read the IR file, restrict to the cohort, and build the full analytic dataset."""
    ir_path = ir_path or IR_PATH
    ir, meta = pyreadstat.read_dta(ir_path)
    ir.columns = [c.lower() for c in ir.columns]

    b19 = num(_col(ir, "b19_01"))
    bidx = num(_col(ir, "bidx_01"))
    eligible = b19.notna() & bidx.notna() & (bidx == 1) & (b19 < REFERENCE_MONTHS)
    sample_flow = pd.DataFrame({
        "step": ["All interviewed women in IR file",
                 "Most recent birth record identified (bidx_01 = 1)",
                 f"Most recent live birth < {REFERENCE_MONTHS} completed months before interview"],
        "n": [len(ir), int(((bidx == 1) & bidx.notna()).sum()), int(eligible.sum())],
    })
    d = ir[eligible].copy().reset_index(drop=True)

    # design + covariates
    d["wt"] = num(d["v005"]) / 1e6
    d["psu"] = num(d["v021"])
    d["strata"] = num(d["v022"])
    d["age_years"] = num(_col(d, "v012"))
    d["parity"] = num(_col(d, "v201"))
    d["anc_visits"] = num(_col(d, "m14_1"))

    # 15 sub-regions from v022
    v22 = num(d["v022"])
    bins = {1: "Kampala", 2: "South Buganda", 3: "South Buganda", 4: "North Buganda", 5: "North Buganda",
            6: "Busoga", 7: "Busoga", 8: "Bukedi", 9: "Bukedi", 10: "Bugisu", 11: "Bugisu",
            12: "Teso", 13: "Teso", 14: "Karamoja", 15: "Karamoja", 16: "Lango", 17: "Lango",
            18: "Acholi", 19: "Acholi", 20: "West Nile", 21: "West Nile", 22: "Bunyoro", 23: "Bunyoro",
            24: "Tooro", 25: "Tooro", 26: "Ankole", 27: "Ankole", 28: "Kigezi", 29: "Kigezi"}
    d["subregion15"] = v22.map(bins)
    d["subregion15"] = pd.Categorical(d["subregion15"], categories=SUBREGION_LEVELS, ordered=True)
    d["subregion15_code"] = d["subregion15"].cat.codes.replace(-1, np.nan) + 1

    # equity / covariate factors (DHS coding)
    d["age_group4"] = pd.cut(d["age_years"], bins=[14, 19, 24, 34, 49],
                             labels=["15-19", "20-24", "25-34", "35-49"])
    edu = num(d["v106"])
    d["education4"] = edu.map({0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"})
    wl = num(d["v190"])
    d["wealth5"] = wl.map({1: "Lowest", 2: "Second", 3: "Middle", 4: "Fourth", 5: "Highest"})
    res = num(d["v025"])
    d["residence2"] = res.map({1: "Urban", 2: "Rural"})
    d["parity3"] = np.select([d["parity"] == 1, d["parity"].between(2, 4), d["parity"] >= 5],
                             ["1", "2-4", "5+"], default=None)
    mar = num(_col(d, "v501"))
    d["marital3"] = np.select(
        [mar.isin([1, 2]), mar == 0, mar.isin([3, 4, 5])],
        ["Married/living together", "Never in union", "Previously married/not in union"],
        default=None)

    # ---- ANC content items
    # BP/urine/blood are ANC-contact content items: no ANC contact -> structural 0.
    d["anc_contact"] = np.select(
        [d["anc_visits"] == 0, (d["anc_visits"] > 0) & (d["anc_visits"] < 98)],
        [0.0, 1.0], default=np.nan)
    for item, src in [("anc_bp", "m42c_1"), ("anc_urine", "m42d_1"), ("anc_blood", "m42e_1")]:
        raw = yes_no(d[src])
        v = pd.Series(np.nan, index=d.index)
        v[d["anc_contact"] == 0] = 0.0
        m1 = d["anc_contact"] == 1
        v[m1] = raw[m1]
        d[item] = v
    # Tetanus, iron, SP/Fansidar, deworming use their own reported values (not forced to 0).
    d["anc_iron"] = yes_no(d["m45_1"])
    d["anc_iptp"] = yes_no(d["m49a_1"])
    d["anc_deworm"] = yes_no(d["m60_1"])

    tt_c, tt_p, tt_y = num(d["m1_1"]), num(d["m1a_1"]), num(d["m1d_1"])
    tet = pd.Series(np.nan, index=d.index)
    tet[(tt_c >= 2) & (tt_c <= 7)] = 1.0
    tet[(tt_c == 1) & (tt_p >= 1) & (tt_p <= 7)] = 1.0
    tet[(tt_c == 0) & (tt_p >= 5) & (tt_p <= 7)] = 1.0
    tet[(tt_c == 0) & (tt_p >= 4) & (tt_p <= 7) & (tt_y < 10)] = 1.0
    tet[(tt_c == 0) & (tt_p >= 3) & (tt_p <= 7) & (tt_y < 5)] = 1.0
    tet[(tt_c == 0) & (tt_p >= 2) & (tt_p <= 7) & (tt_y < 3)] = 1.0
    tet[(tt_c == 0) & (tt_p == 0)] = 0.0
    tet[(tt_c == 1) & (tt_p == 0)] = 0.0
    unprot = (tt_c.isin([0, 1]) & tt_p.notna() &
              ((tt_p <= 1) | ((tt_p >= 2) & (tt_p < 5) & tt_y.notna())))
    tet[unprot & tet.isna()] = 0.0
    d["anc_tetanus"] = tet

    # ---- Delivery / immediate newborn items
    m3a, m3b, m3c = num(d["m3a_1"]), num(d["m3b_1"]), num(d["m3c_1"])
    sba = pd.Series(np.nan, index=d.index)
    sba[(m3a == 1) | (m3b == 1) | (m3c == 1)] = 1.0
    sba[sba.isna() & (m3a.notna() | m3b.notna() | m3c.notna())] = 0.0
    d["delivery_sba"] = sba

    m15 = num(d["m15_1"])
    fac = pd.Series(np.nan, index=d.index)
    fac[m15.isin(FACILITY_CODES)] = 1.0
    fac[m15.isin(HOME_CODES + OTHER_PLACE_CODES)] = 0.0
    d["delivery_facility"] = fac

    for raw_item, src in [("nc_cord", "m78a_1"), ("nc_temp", "m78b_1"), ("nc_bf", "m78e_1")]:
        r = yes_no(d[src])
        v = r.copy()
        v[r.isna() & (d["delivery_facility"] == 0)] = 0.0
        d[raw_item] = v

    # ---- PNC items
    d["mother_pnc_2d"] = dhs_pnc_2days(d["m62_1"], d["m63_1"], _col(d, "m64_1"),
                                       d["m66_1"], d["m67_1"], _col(d, "m68_1"))
    d["newborn_pnc_2d"] = dhs_pnc_2days(d["m74_1"], _col(d, "m75_1"), _col(d, "m76_1"),
                                        d["m70_1"], _col(d, "m71_1"), _col(d, "m72_1"))
    for item, src in [("pnc_danger_counsel", "q473c_1"), ("pnc_bf_counsel", "q473d_1")]:
        r = yes_no(d[src])
        v = r.copy()
        v[r.isna() & (d["mother_pnc_2d"] == 0)] = 0.0
        d[item] = v

    # ---- domain + composite scores
    d["anc_score_fixed"] = row_score_fixed(d, ANC_ITEMS)
    d["delivery_score_fixed"] = row_score_fixed(d, DELIVERY_ITEMS)
    d["pnc_score_fixed"] = row_score_fixed(d, PNC_ITEMS)
    d["anc_score_available"] = row_score_available(d, ANC_ITEMS, MIN_ITEMS_ANC_AVAILABLE)
    d["delivery_score_available"] = row_score_available(d, DELIVERY_ITEMS, MIN_ITEMS_DELIVERY_AVAILABLE)
    d["pnc_score_available"] = row_score_available(d, PNC_ITEMS, MIN_ITEMS_PNC_AVAILABLE)

    fixed_ok = d[["anc_score_fixed", "delivery_score_fixed", "pnc_score_fixed"]].notna().all(axis=1)
    d["eci"] = np.where(fixed_ok,
                        d[["anc_score_fixed", "delivery_score_fixed", "pnc_score_fixed"]].mean(axis=1),
                        np.nan)
    avail_ok = d[["anc_score_available", "delivery_score_available", "pnc_score_available"]].notna().all(axis=1)
    d["eci_available"] = np.where(avail_ok,
                                  d[["anc_score_available", "delivery_score_available", "pnc_score_available"]].mean(axis=1),
                                  np.nan)
    for thr in ECI_THRESHOLDS:
        nm = f"eci_inadequate_{int(thr * 100)}"
        d[nm] = np.where(pd.notna(d["eci"]), (d["eci"] < thr).astype(float), np.nan)

    return d, sample_flow


def load_analytic(path=None):
    path = path or ANALYTIC_PARQUET
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Analytic dataset not found:\n  {path}\nRun phase1_eci_scoring.py first.")
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------- SURVEY ESTIMATION
def _taylor_mean(sub, var):
    """Taylor-linearisation mean of a 0/1 or continuous var on one design subset.

    Replicates R survey::svymean on svydesign(ids=~psu, strata=~strata,
    weights=~wt, nest=TRUE): ratio-estimator linearisation, with-replacement
    PSU variance, and survey.lonely.psu='adjust' (single-PSU strata contribute 0).
    """
    from scipy import stats
    w = sub["wt"].to_numpy(dtype=float)
    y = sub[var].to_numpy(dtype=float)
    wsum = w.sum()
    theta = float((w * y).sum() / wsum)
    u = w * (y - theta) / wsum
    t = pd.DataFrame({"h": sub["strata"].to_numpy(), "p": sub["psu"].to_numpy(), "u": u})
    psu_tot = t.groupby(["h", "p"], sort=False)["u"].sum().reset_index()
    var_est = 0.0
    for _, g in psu_tot.groupby("h"):
        m = len(g)
        if m > 1:                       # lonely PSU (m==1) contributes 0 ('adjust')
            var_est += m / (m - 1) * ((g["u"] - g["u"].mean()) ** 2).sum()
    se = float(np.sqrt(var_est))
    n_psu = psu_tot.shape[0]
    n_str = psu_tot["h"].nunique()
    ddf = max(n_psu - n_str, 1)
    tc = float(stats.t.ppf(0.975, ddf))
    return theta, se, theta - tc * se, theta + tc * se


def svy_mean(df, var, domain=None):
    """Survey-weighted mean/proportion with 95% CI (Taylor linearisation).

    Returns a DataFrame with estimate, se, ci_low, ci_high — one row overall, or
    one row per domain level. Each estimate is computed on the subset with the
    variable observed (and, for a domain, that domain's rows), matching the R
    svymean_each / svyby convention of subsetting the design.
    """
    base = df[df["psu"].notna() & df["strata"].notna() &
              df["wt"].notna() & (df["wt"] > 0) & df[var].notna()].copy()
    base[var] = base[var].astype(float)
    if domain is None:
        est, se, lo, hi = _taylor_mean(base, var)
        return pd.DataFrame([{"estimate": est, "se": se, "ci_low": lo, "ci_high": hi}])
    rows = []
    sub = base[base[domain].notna()]
    for lvl in pd.unique(sub[domain]):
        g = sub[sub[domain] == lvl]
        est, se, lo, hi = _taylor_mean(g, var)
        rows.append({domain: lvl, "estimate": est, "se": se, "ci_low": lo,
                     "ci_high": hi, "n": len(g)})
    return pd.DataFrame(rows)


def pct(x, d=1):
    try:
        return round(float(x) * 100, d)
    except Exception:
        return np.nan


def ci_str(est, lo, hi, d=1, as_pct=True, sign=False):
    try:
        f = 100.0 if as_pct else 1.0
        s = "+" if (sign and est * f > 0) else ""
        sep = " to " if sign else "–"
        return f"{s}{est*f:.{d}f} ({lo*f:.{d}f}{sep}{hi*f:.{d}f})"
    except Exception:
        return ""


def write_xlsx(path, sheets):
    """Write {sheet_name: DataFrame} to a styled .xlsx (blue header, frozen row)."""
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            fill = PatternFill("solid", fgColor="17365D")
            for j, col in enumerate(df.columns, start=1):
                c = ws.cell(row=1, column=j)
                c.font = Font(bold=True, color="FFFFFF"); c.fill = fill
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                width = min(max(len(str(col)) + 2,
                                int(df[col].astype(str).str.len().max() if len(df) else 10) + 2), 48)
                ws.column_dimensions[get_column_letter(j)].width = width
            ws.freeze_panes = "A2"
    return path
