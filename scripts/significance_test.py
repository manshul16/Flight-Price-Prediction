"""
Paired flight-level bootstrap test of holdout MAE differences versus LightGBM.

Uses the models saved in models/ by scripts/run_all.py (run that first) and
recreates the identical grouped holdout split. Writes
results/tables/significance_tests.csv.

Usage (from the repository root):
    python scripts/significance_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.features import engineer_features, get_feature_column_names  # noqa: E402

N_BOOTSTRAP = 2000
REFERENCE = "lightgbm"
CANDIDATES = [
    "stacking_ensemble",
    "histgradientboosting",
    "extratrees",
    "randomforest",
    "catboost",
    "ridge",
    "elasticnet",
]


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    df = engineer_features(load_dataset(config))
    cat_cols, num_cols, bin_cols = get_feature_column_names(config)
    feature_cols = cat_cols + num_cols + bin_cols

    groups = df[config.group_column].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.holdout_fraction, random_state=config.seed)
    _, test_idx = next(splitter.split(df, groups=groups))
    test = df.iloc[test_idx].reset_index(drop=True)
    X_test = test[feature_cols]
    y_true = np.expm1(test[config.log_target_column].to_numpy())
    test_groups = test[config.group_column].to_numpy()

    abs_err = {}
    for name in [REFERENCE] + CANDIDATES:
        model = joblib.load(config.models_dir / f"{name}.joblib")
        pred = np.clip(np.expm1(model.predict(X_test)), 0, None)
        abs_err[name] = np.abs(y_true - pred)

    # Resample whole flights (groups) because rows of the same flight are correlated.
    unique_groups, inverse = np.unique(test_groups, return_inverse=True)
    rows_per_group = np.bincount(inverse)
    rng = np.random.default_rng(config.seed)
    boot_idx = rng.integers(0, len(unique_groups), size=(N_BOOTSTRAP, len(unique_groups)))

    records = []
    for name in CANDIDATES:
        diff = abs_err[name] - abs_err[REFERENCE]
        diff_per_group = np.bincount(inverse, weights=diff)
        boots = diff_per_group[boot_idx].sum(axis=1) / rows_per_group[boot_idx].sum(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_value = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / N_BOOTSTRAP)
        records.append(
            {
                "Model": name,
                "Reference": REFERENCE,
                "Test_MAE": round(float(abs_err[name].mean()), 2),
                "Delta_MAE_vs_Reference": round(float(diff.mean()), 2),
                "CI95_Low": round(float(lo), 2),
                "CI95_High": round(float(hi), 2),
                "Bootstrap_P_Value": round(float(p_value), 4),
                "Significant_at_0.05": bool(p_value < 0.05),
            }
        )

    out = pd.DataFrame(records)
    out_path = config.tables_dir / "significance_tests.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
