# -*- coding: utf-8 -*-
"""Run the full ECI Python pipeline (Phase 1 -> 5) in order."""
import importlib

PHASES = [
    ("phase1_eci_scoring", "Phase 1 — ECI scoring + survey-weighted descriptives"),
    ("phase2_validation", "Phase 2 — validation & robustness"),
    ("phase3_survey_models", "Phase 3 — survey-weighted adjusted models"),
    ("phase4_multilevel", "Phase 4 — multilevel variance decomposition"),
    ("phase5_spatial", "Phase 5 — spatial (choropleths + areal Moran's I/LISA)"),
]

if __name__ == "__main__":
    for mod, title in PHASES:
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
        try:
            importlib.import_module(mod).main()
        except Exception as e:
            print(f"!! {mod} failed: {e}")
            if mod == "phase5_spatial":
                print("   (Phase 5 needs the parish .shp — set ECI_PARISH_SHP. "
                      "Phases 1-4 do not need it.)")
            else:
                raise
