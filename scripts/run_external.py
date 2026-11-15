"""
H5 experiment: do external parameters (calendar, weather, competitor fares) improve
flight price prediction?

Uses exactly the same grouped holdout split and LightGBM configuration as the
feature ablation in scripts/run_all.py, so the "Itinerary only" row reproduces
the "Full Feature Set" row of results/tables/ablation_study.csv.

Outputs
    results/tables/external_ablation.csv
    results/tables/external_feature_importance.csv
    results/tables/external_intervals.csv
    results/figures/external_ablation.png
    results/figures/external_shap_bar.png

Usage (from the repository root; first run downloads ~6 small weather files):
    python scripts/run_external.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import yaml  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.external_features import (  # noqa: E402
    CALENDAR_FEATURES,
    CLIMATE_FEATURES,
    COMPETITOR_FEATURES,
    ORACLE_FEATURES,
    add_external_features,
)
from src.features import build_sklearn_preprocessor, engineer_features  # noqa: E402
from src.intervals import ConformalizedQuantileRegressor, train_quantile_lightgbm  # noqa: E402
from src.utils import evaluate_predictions, set_seed, setup_logger  # noqa: E402

logger = setup_logger("run_external")

N_BOOT = 2000
BUDGET_CARRIERS = ["AirAsia", "SpiceJet"]
BINARY_EXTERNAL = {"is_weekend_departure", "is_holiday_window"}

# Colors: validated categorical palette, fixed slot order (dataviz reference palette).
GROUP_COLORS = {
    "Itinerary": "#2a78d6",
    "Calendar": "#eb6834",
    "Weather": "#1baf7a",
    "Competitor": "#eda100",
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def feature_lists(config, extra: list[str]):
    cat = [c for c in config.categorical_features if c != "route_target_encoded"]
    num = [c for c in config.numerical_features if c != "route_target_encoded"]
    binc = list(config.binary_features)
    num = num + [c for c in extra if c not in BINARY_EXTERNAL]
    binc = binc + [c for c in extra if c in BINARY_EXTERNAL]
    return cat, num, binc


def fit_predict(config, X_tr, y_tr, X_te, extra):
    cat, num, binc = feature_lists(config, extra)
    pre = build_sklearn_preprocessor(cat, num, binc, use_one_hot=False)
    A_tr = pre.fit_transform(X_tr)
    A_te = pre.transform(X_te)
    model = LGBMRegressor(
        n_estimators=150, learning_rate=0.08, num_leaves=63,
        random_state=config.seed, n_jobs=4, verbose=-1,
    )
    model.fit(A_tr, y_tr)
    return model, pre, A_te, num + cat + binc, model.predict(A_te)


def paired_bootstrap(err_a, err_b, groups, seed):
    """MAE(b) - MAE(a), resampling whole flights."""
    uniq, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    s = np.bincount(inv, weights=err_b - err_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boots = s[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / N_BOOT)
    return float((err_b - err_a).mean()), float(lo), float(hi), float(p)


def feature_group(name: str) -> str:
    if name in CALENDAR_FEATURES:
        return "Calendar"
    if name in CLIMATE_FEATURES or name in ORACLE_FEATURES:
        return "Weather"
    if name in COMPETITOR_FEATURES:
        return "Competitor"
    return "Itinerary"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_ablation(table: pd.DataFrame, out: Path):
    rows = table[table["Feature_Set"] != "Itinerary only"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    y = np.arange(len(rows))
    delta = rows["Delta_MAE"].to_numpy()
    lo = delta - rows["CI95_Low"].to_numpy()
    hi = rows["CI95_High"].to_numpy() - delta
    oracle = rows["Feature_Set"].str.contains("oracle").to_numpy()
    for yi, d, is_oracle in zip(y, delta, oracle):
        ax.barh(yi, d, height=0.5, color="#2a78d6", alpha=0.45 if is_oracle else 1.0,
                hatch="///" if is_oracle else None, edgecolor=SURFACE, lw=0)
    ax.errorbar(delta, y, xerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["Feature_Set"], color=INK, fontsize=9)
    for yi, d, p, ci_hi in zip(y, delta, rows["Bootstrap_P_Value"], rows["CI95_High"]):
        label = f"{d:+,.0f} INR, p = {p:.3f}" + ("" if p < 0.05 else " (n.s.)")
        ax.text(max(ci_hi, 0) + 8, yi, label, va="center", ha="left", fontsize=8.5, color=INK)
    left = min(rows["CI95_Low"].min(), 0) - 20
    right = max(rows["CI95_High"].max(), 0) + 150
    ax.set_xlim(left, right)
    ax.set_xlabel("Change in holdout MAE vs itinerary-only model (INR, 95% flight-bootstrap CI)",
                  color=INK_2, fontsize=9)
    ax.set_title("Effect of adding external parameters (lower is better)", color=INK,
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_shap(imp: pd.DataFrame, out: Path, top: int = 20):
    data = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    colors = [GROUP_COLORS[g] for g in data["Group"]]
    ax.barh(np.arange(len(data)), data["Mean_Abs_SHAP"], height=0.62, color=colors, edgecolor=SURFACE, lw=1)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["Feature"], fontsize=8.5, color=INK)
    ax.set_xlabel("Mean |SHAP value| (log-price units)", color=INK_2, fontsize=9)
    ax.set_title(f"Top {top} features of the model with all external parameters", color=INK,
                 fontsize=11, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(), frameon=False, loc="lower right", fontsize=8.5,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    raw_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    set_seed(config.seed)

    df = engineer_features(load_dataset(config))
    df = add_external_features(df, raw_cfg, PROJECT_ROOT)

    groups = df[config.group_column].to_numpy()
    split = GroupShuffleSplit(n_splits=1, test_size=config.holdout_fraction, random_state=config.seed)
    tr_idx, te_idx = next(split.split(df, groups=groups))
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    y_tr = train[config.log_target_column].to_numpy()
    y_te = test[config.log_target_column].to_numpy()
    y_te_real = np.expm1(y_te)
    g_te = test[config.group_column].to_numpy()
    logger.info("Train %d rows, test %d rows", len(train), len(test))

    variants = [
        ("Itinerary only", []),
        ("+ Calendar & holidays", CALENDAR_FEATURES),
        ("+ Weather (climate normals)", CLIMATE_FEATURES),
        ("+ Competitor fares", COMPETITOR_FEATURES),
        ("All external parameters", CALENDAR_FEATURES + CLIMATE_FEATURES + COMPETITOR_FEATURES),
        ("All + observed weather (oracle, leaky)",
         CALENDAR_FEATURES + CLIMATE_FEATURES + COMPETITOR_FEATURES + ORACLE_FEATURES),
    ]

    last3 = test["days_left"].to_numpy() <= 3
    budget = test["airline"].isin(BUDGET_CARRIERS).to_numpy()
    records, errors, fitted = [], {}, {}
    for name, extra in variants:
        logger.info("Fitting variant: %s", name)
        model, pre, A_te, names, pred = fit_predict(config, train, y_tr, test, extra)
        pred_real = np.clip(np.expm1(pred), 0, None)
        err = np.abs(y_te_real - pred_real)
        errors[name] = err
        fitted[name] = (model, pre, A_te, names)
        m = evaluate_predictions(y_te, pred, is_log_target=True)
        budget_mape = float(np.mean(err[budget] / y_te_real[budget]) * 100)
        records.append({
            "Feature_Set": name,
            "N_Extra_Features": len(extra),
            "Test_MAE": m["MAE"], "Test_RMSE": m["RMSE"], "Test_R2": m["R2"],
            "Test_MAPE": m["MAPE"], "Test_sMAPE": m["sMAPE"],
            "MAE_0_3_days": round(float(err[last3].mean()), 2),
            "MAPE_budget_carriers": round(budget_mape, 2),
        })

    base_err = errors["Itinerary only"]
    for rec in records:
        if rec["Feature_Set"] == "Itinerary only":
            rec.update({"Delta_MAE": 0.0, "CI95_Low": 0.0, "CI95_High": 0.0, "Bootstrap_P_Value": np.nan})
            continue
        d, lo, hi, p = paired_bootstrap(base_err, errors[rec["Feature_Set"]], g_te, config.seed)
        rec.update({"Delta_MAE": round(d, 2), "CI95_Low": round(lo, 2), "CI95_High": round(hi, 2),
                    "Bootstrap_P_Value": round(p, 4)})
    table = pd.DataFrame(records)
    tables, figures = config.tables_dir, config.figures_dir
    table.to_csv(tables / "external_ablation.csv", index=False)
    print(table.to_string(index=False))

    # SHAP on the model with all (non-oracle) external parameters
    model, pre, A_te, names = fitted["All external parameters"]
    rng = np.random.RandomState(config.seed)
    sample = A_te[rng.choice(len(A_te), size=min(2000, len(A_te)), replace=False)]
    sv = shap.TreeExplainer(model).shap_values(sample)
    imp = pd.DataFrame({"Feature": names, "Mean_Abs_SHAP": np.abs(sv).mean(axis=0)})
    imp["Group"] = imp["Feature"].map(feature_group)
    imp = imp.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    imp["Rank"] = np.arange(1, len(imp) + 1)
    imp["Mean_Abs_SHAP"] = imp["Mean_Abs_SHAP"].round(5)
    imp.to_csv(tables / "external_feature_importance.csv", index=False)
    print(imp.head(15).to_string(index=False))

    # Prediction intervals: itinerary-only vs all external parameters (CQR, same halves as run_all)
    cal_i, ev_i = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=config.seed)
                       .split(test, groups=g_te))
    cal, ev = test.iloc[cal_i], test.iloc[ev_i]
    rows = []
    for name in ("Itinerary only", "All external parameters"):
        extra = dict(variants)[name]
        cat, num, binc = feature_lists(config, extra)
        pre_q = build_sklearn_preprocessor(cat, num, binc, use_one_hot=False)
        q = train_quantile_lightgbm(pre_q.fit_transform(train), y_tr, [0.10, 0.90], seed=config.seed)
        Xc, Xe = pre_q.transform(cal), pre_q.transform(ev)
        cqr = ConformalizedQuantileRegressor(config.target_coverage)
        cqr.calibrate(np.expm1(cal[config.log_target_column].to_numpy()),
                      np.expm1(q[0.10].predict(Xc)), np.expm1(q[0.90].predict(Xc)))
        y_ev = np.expm1(ev[config.log_target_column].to_numpy())
        for calibrated in (False, True):
            lo_b, hi_b = cqr.predict_intervals(np.expm1(q[0.10].predict(Xe)), np.expm1(q[0.90].predict(Xe)),
                                               apply_calibration=calibrated)
            rows.append({
                "Feature_Set": name,
                "CQR_Calibrated": "Yes" if calibrated else "No",
                "Empirical_PICP_%": round(float(np.mean((y_ev >= lo_b) & (y_ev <= hi_b)) * 100), 2),
                "MPIW_INR": round(float(np.mean(hi_b - lo_b)), 2),
                "CQR_Offset_INR": round(cqr.conformal_offset_, 2) if calibrated else 0.0,
            })
    intervals = pd.DataFrame(rows)
    intervals.to_csv(tables / "external_intervals.csv", index=False)
    print(intervals.to_string(index=False))

    plot_ablation(table, figures / "external_ablation.png")
    plot_shap(imp, figures / "external_shap_bar.png")
    logger.info("H5 experiment complete.")


if __name__ == "__main__":
    main()
