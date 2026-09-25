"""Side-by-side montage of the complexity-sweep panels: ICU (139 subsets) vs. census (217 log-income
and 183 insurance cells). Rows = the four knobs; missing panels (sweep not finished) are left blank.

Run from anywhere:  python census/py/xx_montage_icu_vs_census.py
Output: census/out_py/figures/complexity_icu_vs_census.png / .pdf
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents] if (p / "obd_engine").exists())
ICU = ROOT / "Real-data example" / "Figures"
CEN = ROOT / "census" / "out_py" / "figures"
ROWS = [("xgb_depth", "XGBoost: tree depth"), ("xgb_trees", "XGBoost: number of trees"),
        ("nn_width", "Neural net: width (3 hidden layers)"), ("nn_depth", "Neural net: depth (width 32)")]
COLS = [("ICU, 139 subsets (Table 1)", lambda k: ICU / f"complexity_{k}.png"),
        ("Census, log income, 217 cells (Table 8)", lambda k: CEN / f"census_complexity_{k}_pincp.png"),
        ("Census, health insurance, 183 cells (Table 9)", lambda k: CEN / f"census_complexity_{k}_hicov.png")]

fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(8 * len(COLS), 6.4 * len(ROWS)))
for i, (knob, rlab) in enumerate(ROWS):
    for j, (clab, path_of) in enumerate(COLS):
        ax = axes[i, j]; ax.axis("off")
        p = path_of(knob)
        if p.exists():
            ax.imshow(mpimg.imread(p))
        else:
            ax.text(0.5, 0.5, "sweep still running", ha="center", va="center", fontsize=22, color="0.5", transform=ax.transAxes)
        if i == 0: ax.set_title(clab, fontsize=24, pad=14)
        if j == 0: ax.text(-0.02, 0.5, rlab, rotation=90, ha="right", va="center", fontsize=24, transform=ax.transAxes)
fig.tight_layout()
CEN.mkdir(parents=True, exist_ok=True)
fig.savefig(CEN / "complexity_icu_vs_census.png", dpi=70, bbox_inches="tight")
fig.savefig(CEN / "complexity_icu_vs_census.pdf", bbox_inches="tight")
print("wrote", CEN / "complexity_icu_vs_census.png")
