"""Render the pipeline flowchart (Section 2.2) as a PNG for the Word version of the report."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "pipeline_diagram.png"

BOX_W, BOX_H = 2.5, 0.78
RES_W = 3.3  # wider box for the final results node
FILL = {"data": "#E8F0FB", "model": "#EAF6EC", "interval": "#FDF1E3", "out": "#F1ECF8"}
EDGE = {"data": "#2F5D9E", "model": "#2E7D45", "interval": "#B5651D", "out": "#6A4C9C"}

# name: (x_center, y_center, label, kind)
NODES = {
    "raw": (1.6, 7.0, "Raw CSV\n300,153 rows", "data"),
    "clean": (4.6, 7.0, "Clean + validate", "data"),
    "feat": (7.6, 7.0, "Feature engineering\n(+6 features)", "data"),
    "split": (10.6, 7.0, "Group split by flight\n(GroupShuffleSplit)", "data"),
    "train": (7.6, 5.4, "Train set\n245,189 rows / 1,248 flights", "data"),
    "hold": (13.0, 5.4, "Holdout\n54,964 rows / 313 flights", "data"),
    "tune": (4.6, 5.4, "Optuna tuning\n3-fold GroupKFold", "model"),
    "cv": (4.6, 3.8, "5-fold GroupKFold CV\n9 models + OOF preds", "model"),
    "stack": (4.6, 2.2, "Ridge meta-learner\n(stacking)", "model"),
    "qr": (7.6, 3.8, "Quantile LightGBM\nτ = 0.1, 0.9", "interval"),
    "cal": (10.6, 3.8, "Calibration half\n24,611 rows", "interval"),
    "cqr": (9.1, 2.2, "CQR offset Q", "interval"),
    "ev": (13.0, 2.2, "Interval test half\n30,353 rows → PICP, MPIW", "interval"),
    "res": (7.6, 0.6, "Holdout metrics, SHAP,\nerror breakdowns, timing curves", "out"),
}

EDGES = [
    ("raw", "clean", ""), ("clean", "feat", ""), ("feat", "split", ""),
    ("split", "train", "80% of flights"), ("split", "hold", "20% of flights"),
    ("train", "tune", ""), ("tune", "cv", ""), ("cv", "stack", ""),
    ("train", "qr", ""), ("hold", "cal", "50%"), ("hold", "ev", "50%"),
    ("qr", "cqr", ""), ("cal", "cqr", ""), ("cqr", "ev", ""),
    ("stack", "res", ""), ("hold", "res", ""),
]


def anchor(name, toward):
    """Point on the box border of `name` facing node `toward`."""
    x, y = NODES[name][:2]
    tx, ty = NODES[toward][:2]
    dx, dy = tx - x, ty - y
    if abs(dx) * BOX_H > abs(dy) * BOX_W:
        return (x + (BOX_W / 2) * (1 if dx > 0 else -1), y)
    return (x, y + (BOX_H / 2) * (1 if dy > 0 else -1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=200)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(-0.1, 7.6)
    ax.axis("off")

    for src, dst, label in EDGES:
        if (src, dst) == ("hold", "res"):
            # route down the right margin so the line doesn't cross other boxes
            hx, hy = NODES["hold"][0] + BOX_W / 2, NODES["hold"][1]
            rail = hx + 0.15
            ax.plot([hx, rail, rail], [hy, hy, 0.6], color="#555", lw=1.1, zorder=1)
            ax.add_patch(FancyArrowPatch((rail, 0.6), (NODES["res"][0] + RES_W / 2, 0.6),
                                         arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.1,
                                         shrinkA=0, shrinkB=0))
            continue
        a, b = anchor(src, dst), anchor(dst, src)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.1,
                                     shrinkA=0, shrinkB=0, zorder=1))
        if label:
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ax.text(mx, my, label, fontsize=8, ha="center", va="center", color="#333",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=3)

    for key, (x, y, text, kind) in NODES.items():
        w = RES_W if key == "res" else BOX_W
        ax.add_patch(FancyBboxPatch((x - w / 2, y - BOX_H / 2), w, BOX_H,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=FILL[kind], ec=EDGE[kind], lw=1.4, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color="#1a1a1a", zorder=4)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
