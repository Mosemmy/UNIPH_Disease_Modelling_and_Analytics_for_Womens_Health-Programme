# -*- coding: utf-8 -*-
"""
PHASE 5 — spatial analysis (Python port of R Phase 5 + Phase 5b).

MODULE A: survey-weighted sub-region choropleths (composite ECI, ECI<0.70, and
          the ANC / delivery-newborn / PNC domains), boundaries aggregated from
          the UBOS 2024 parish polygons to the 15 UDHS sampling regions.
MODULE B: areal Global Moran's I + exploratory Local Moran (LISA) on the 15
          regions (queen contiguity, permutation inference) + Moran scatterplot.

UDHS 2022 has no released DHS GPS cluster layer, so cluster-level hotspots
(Getis-Ord Gi*, SaTScan) and facility-distance analyses are out of scope.
With 15 areal units the Moran tests are confirmatory/low-power; the primary
clustering evidence is the Phase 4 multilevel ICC/MOR.

Requires the parish shapefile (same one the R pipeline uses):
  <DATA_DIR>/uganda_parishes_2024/uganda_parishes_2024.shp   (override: ECI_PARISH_SHP)

Run:  python phase5_spatial.py
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eci_common as ec

PARISH_SHP = os.environ.get(
    "ECI_PARISH_SHP",
    os.path.join(ec.DATA_DIR, "uganda_parishes_2024", "uganda_parishes_2024.shp"))
CHARTS = os.path.join(ec.OUTPUT_DIR, "Charts")
TABLES = os.path.join(ec.OUTPUT_DIR, "Tables")
NSIM = 9999

# ---- district -> 15-region crosswalk (identical to the R Phase 5 mapping)
SOUTH_BUGANDA = ["Butambala", "Gomba", "Mpigi", "Bukomansimbi", "Kalangala", "Kalungu",
                 "Kyotera", "Lwengo", "Lyantonde", "Masaka", "Rakai", "Sembabule", "Ssembabule", "Wakiso"]
NORTH_BUGANDA = ["Buikwe", "Buvuma", "Kassanda", "Kasanda", "Kayunga", "Kiboga", "Kyankwanzi",
                 "Luwero", "Luweero", "Mityana", "Mubende", "Mukono", "Nakaseke", "Nakasongola"]
REGION_DISTRICTS = {
    "Kampala": ["Kampala"], "South Buganda": SOUTH_BUGANDA, "North Buganda": NORTH_BUGANDA,
    "Busoga": ["Bugiri", "Bugweri", "Buyende", "Iganga", "Jinja", "Kaliro", "Kamuli", "Luuka", "Mayuge", "Namayingo", "Namutumba"],
    "Bukedi": ["Budaka", "Busia", "Butaleja", "Butebo", "Kibuku", "Pallisa", "Tororo"],
    "Bugisu": ["Bududa", "Bukwo", "Bulambuli", "Kapchorwa", "Kween", "Manafwa", "Mbale", "Namisindwa", "Sironko"],
    "Teso": ["Amuria", "Bukedea", "Kaberamaido", "Kalaki", "Kapelebyong", "Katakwi", "Kumi", "Ngora", "Serere", "Soroti"],
    "Karamoja": ["Abim", "Amudat", "Kaabong", "Karenga", "Kotido", "Moroto", "Nabilatuk", "Nakapiripirit", "Napak"],
    "Lango": ["Alebtong", "Amolatar", "Apac", "Dokolo", "Kole", "Kwania", "Lira", "Otuke", "Oyam"],
    "Acholi": ["Agago", "Amuru", "Gulu", "Kitgum", "Lamwo", "Nwoya", "Omoro", "Pader"],
    "West Nile": ["Adjumani", "Arua", "Koboko", "Madi Okollo", "Maracha", "Moyo", "Nebbi", "Obongi", "Pakwach", "Terego", "Yumbe", "Zombo"],
    "Bunyoro": ["Buliisa", "Hoima", "Kagadi", "Kakumiro", "Kibaale", "Kikuube", "Kiryandongo", "Masindi"],
    "Tooro": ["Bundibugyo", "Bunyangabu", "Fort Portal", "Kabarole", "Kamwenge", "Kasese", "Kitagwenda", "Kyegegwa", "Kyenjojo", "Ntoroko"],
    "Ankole": ["Buhweju", "Bushenyi", "Ibanda", "Isingiro", "Kazo", "Kiruhura", "Mbarara", "Mitooma", "Ntungamo", "Rubirizi", "Rwampara", "Sheema"],
    "Kigezi": ["Kabale", "Kanungu", "Kisoro", "Rubanda", "Rukiga", "Rukungiri"],
}


def norm_admin(s):
    s = re.sub(r"[-_/]+", " ", str(s).upper().strip())
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r" CITY$", "", s)
    s = re.sub(r" DISTRICT$", "", s)
    return s.strip()


def district_to_udhs(district):
    d = norm_admin(district)
    for region, dists in REGION_DISTRICTS.items():
        if d in [norm_admin(x) for x in dists]:
            return region
    if "KAMPALA" in d:
        return "Kampala"
    return None


def subregion_means(d):
    base = ec.svy_mean(d, "eci", domain="subregion15")[["subregion15", "estimate"]].rename(columns={"estimate": "eci"})
    for v in ["eci_inadequate_70", "anc_score_fixed", "delivery_score_fixed", "pnc_score_fixed"]:
        r = ec.svy_mean(d, v, domain="subregion15")[["subregion15", "estimate"]].rename(columns={"estimate": v})
        base = base.merge(r, on="subregion15")
    base["subregion15"] = base["subregion15"].astype(str)
    return base


def build_subregion_gdf(values):
    import geopandas as gpd
    if not os.path.exists(PARISH_SHP):
        raise FileNotFoundError(
            f"Parish shapefile not found:\n  {PARISH_SHP}\n"
            "Set ECI_PARISH_SHP to the UBOS 2024 parish .shp (the one the R pipeline uses).")
    parishes = gpd.read_file(PARISH_SHP)
    if parishes.crs is None:
        parishes.set_crs(4326, inplace=True)
    parishes["geometry"] = parishes.buffer(0)
    dcol = next((c for c in parishes.columns
                 if c.lower() in ("district", "adm2_name")), None)
    if dcol is None:
        raise ValueError("Parish layer has no recognised District field.")
    parishes["subregion15"] = parishes[dcol].map(district_to_udhs)
    unmapped = sorted(parishes.loc[parishes["subregion15"].isna(), dcol].dropna().unique())
    if unmapped:
        print("WARNING: unmapped districts ->", unmapped[:20])
    gdf = (parishes.dropna(subset=["subregion15"])
           .dissolve(by="subregion15").reset_index()
           .merge(values, on="subregion15").to_crs(4326))
    return gdf, unmapped


def _choropleth(gdf, col, title, path, pct=True, cmap="Blues"):
    g = gdf.copy()
    g["_v"] = g[col] * (100 if pct else 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    g.plot(column="_v", cmap=cmap, linewidth=0.2, edgecolor="grey",
           legend=True, ax=ax, legend_kwds={"shrink": 0.6})
    ax.set_title(title, fontsize=12); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def areal_moran(gdf):
    from libpysal.weights import Queen, KNN
    from esda.moran import Moran, Moran_Local
    try:
        w = Queen.from_dataframe(gdf, use_index=False)
        method = "Queen contiguity"
        if min(w.cardinalities.values()) == 0:
            raise ValueError("isolate")
    except Exception:
        w = KNN.from_dataframe(gdf, k=4)
        method = "Centroid k-NN (k=4)"
    w.transform = "r"

    rows = []
    for col, lab in [("eci", "Composite ECI"), ("eci_inadequate_70", "Inadequate coverage (ECI < 0.70)")]:
        mi = Moran(gdf[col].values, w, permutations=NSIM, transformation="r")
        rows.append([lab, len(gdf), round(mi.I, 4), round(mi.EI, 4),
                     round(mi.p_norm, 4), round(mi.p_sim, 4), NSIM, method])
    global_moran = pd.DataFrame(rows, columns=["outcome", "n_regions", "moran_I", "expected_I",
                                               "analytic_p", "permutation_p", "permutations", "weights"])

    lisa = Moran_Local(gdf["eci"].values, w, permutations=NSIM)
    qmap = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    ct = [qmap[q] if p < 0.05 else "Not significant" for q, p in zip(lisa.q, lisa.p_sim)]
    lisa_tbl = pd.DataFrame({
        "subregion15": gdf["subregion15"].values, "eci_pct": (100 * gdf["eci"]).round(1).values,
        "local_I": np.round(lisa.Is, 3), "local_p_sim": np.round(lisa.p_sim, 4),
        "cluster_type": ct}).sort_values("eci_pct", ascending=False)
    return global_moran, lisa_tbl, w


def _moran_scatter(gdf, w, I, path):
    from libpysal.weights.spatial_lag import lag_spatial
    z = (gdf["eci"].values - gdf["eci"].mean()) / gdf["eci"].std(ddof=0)
    lag = lag_spatial(w, z)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, ls="--", color="grey"); ax.axvline(0, ls="--", color="grey")
    ax.scatter(z, lag, color="black")
    xs = np.array([z.min(), z.max()]); ax.plot(xs, I * xs, color="#2166AC", lw=2)
    for xi, yi, lab in zip(z, lag, gdf["subregion15"].values):
        ax.annotate(lab, (xi, yi), fontsize=7, xytext=(2, 3), textcoords="offset points")
    ax.set_xlabel("Standardised sub-region ECI"); ax.set_ylabel("Spatial lag of standardised ECI")
    ax.set_title(f"Moran scatterplot — composite ECI (slope = Global Moran's I = {I:.3f})")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ec.ensure_dirs()
    d = ec.load_analytic()
    values = subregion_means(d)
    gdf, unmapped = build_subregion_gdf(values)
    print(f"Sub-region polygons: {len(gdf)} | unmapped districts: {len(unmapped)}")

    # Module A — choropleths
    _choropleth(gdf, "eci", "Composite effective coverage index by UDHS sub-region",
                os.path.join(CHARTS, "Figure_P5_Composite_ECI_Map.png"))
    _choropleth(gdf, "eci_inadequate_70", "Inadequate coverage (ECI < 0.70) by UDHS sub-region",
                os.path.join(CHARTS, "Figure_P5_Inadequate_Map.png"), cmap="Blues_r")
    for v, lab in [("anc_score_fixed", "ANC"), ("delivery_score_fixed", "Delivery_newborn"),
                   ("pnc_score_fixed", "PNC")]:
        _choropleth(gdf, v, f"{lab} domain score by UDHS sub-region",
                    os.path.join(CHARTS, f"Figure_P5_{lab}_Map.png"))

    # Module B — areal Moran's I + LISA
    global_moran, lisa_tbl, w = areal_moran(gdf)
    I_eci = global_moran.loc[global_moran.outcome == "Composite ECI", "moran_I"].iloc[0]
    _moran_scatter(gdf, w, I_eci, os.path.join(CHARTS, "Figure_P5_Moran_Scatter.png"))

    map_values = values.copy()
    for c in ["eci", "eci_inadequate_70", "anc_score_fixed", "delivery_score_fixed", "pnc_score_fixed"]:
        map_values[c] = (100 * map_values[c]).round(1)

    out = os.path.join(ec.OUTPUT_DIR, "ECI_Python_Phase5_Spatial.xlsx")
    ec.write_xlsx(out, {"Subregion_Map_Values": map_values, "Global_Moran": global_moran,
                        "LISA_Subregions": lisa_tbl})
    print("\nPHASE 5 COMPLETE")
    print(global_moran.to_string(index=False))
    print("LISA cluster types:", lisa_tbl["cluster_type"].value_counts().to_dict())
    print("Workbook:", out, "| Figures in:", CHARTS)


if __name__ == "__main__":
    main()
