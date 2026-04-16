"""LightGBM + SHAP scoring, tiering, and preset scenarios (from what_if_simulator notebook)."""

from __future__ import annotations

import json
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import shap

from what_if_app.config import settings

_booster: lgb.Booster | None = None
_explainer: shap.TreeExplainer | None = None
V1_FEATURES: list[str] = []


def load_model_from_mlflow() -> tuple[lgb.Booster, shap.TreeExplainer]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    mlflow.lightgbm.autolog(disable=True)
    if settings.local_model_path:
        booster = mlflow.lightgbm.load_model(settings.local_model_path)
    else:
        rid = settings.mlflow_run_id.strip()
        booster = mlflow.lightgbm.load_model(f"runs:/{rid}/model")
    explainer = shap.TreeExplainer(booster)
    return booster, explainer


def init_runtime(booster: lgb.Booster, explainer: shap.TreeExplainer) -> None:
    global _booster, _explainer, V1_FEATURES
    _booster = booster
    _explainer = explainer
    V1_FEATURES = list(booster.feature_name())


def is_ready() -> bool:
    return _booster is not None and _explainer is not None


def get_booster() -> lgb.Booster:
    if _booster is None:
        raise RuntimeError("Model not loaded.")
    return _booster


def get_explainer() -> shap.TreeExplainer:
    if _explainer is None:
        raise RuntimeError("Explainer not loaded.")
    return _explainer


TIER_BOUNDARIES = [
    (1, 0.0000, 0.0393),
    (2, 0.0393, 0.0597),
    (3, 0.0597, 0.0787),
    (4, 0.0787, 0.1016),
    (5, 0.1016, 0.1398),
    (6, 0.1398, 0.2129),
    (7, 0.2129, 0.3426),
    (8, 0.3426, 0.5897),
    (9, 0.5897, 0.7855),
    (10, 0.7855, 1.0000),
]

TIER_LABELS = {
    "Good": (0.0000, 0.0597),
    "Okay": (0.0597, 0.5897),
    "Risky": (0.5897, 1.0000),
}


def build_feature_groups(features: list[str]) -> dict[str, list[str]]:
    return {
        "Delinquency": [f for f in features if f.startswith("delinquent_")],
        "Payment Timing": [
            f
            for f in features
            if any(f.startswith(p) for p in ["paid_amt_", "paid_cnt_", "ontime_amt_", "early_amt_", "late_amt_", "late_cnt_"])
        ],
        "Retry Charges": [f for f in features if f.startswith("retry_charge_")],
        "Charge Outcomes": [
            f for f in features if any(f.startswith(p) for p in ["charge_approved_", "charge_declined_"])
        ],
        "Purchase Requests": [f for f in features if f.startswith("pr_")],
        "Orders / Transactions": [
            f for f in features if any(f.startswith(p) for p in ["transact_", "gt4_installments_"])
        ],
        "Card Management": [
            f for f in features if any(f.startswith(p) for p in ["card_", "payment_card_"])
        ],
        "Balance / Credit": [
            f for f in features if any(f.startswith(p) for p in ["credit_utilization_", "outstanding_amt_", "due_amt_"])
        ],
        "Scheduled / Manual": [
            f for f in features if any(f.startswith(p) for p in ["scheduled_charge_", "manual_charge_"])
        ],
        "Tenure": [f for f in features if f.startswith("days_since_")],
    }


def score_to_tier_num(score: float) -> int:
    for tier_num, lo, hi in TIER_BOUNDARIES:
        if lo <= score <= hi:
            return tier_num
    return 10


def score_to_label(score: float) -> str:
    for label, (lo, hi) in TIER_LABELS.items():
        if lo <= score <= hi:
            return label
    return "Risky"


def profile_row_to_df(row: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{f: float(row.get(f, 0.0)) for f in V1_FEATURES}])


def score_profile(profile_df: pd.DataFrame) -> tuple[float, np.ndarray, float]:
    booster = get_booster()
    explainer = get_explainer()
    score = float(booster.predict(profile_df[V1_FEATURES])[0])
    sv = explainer.shap_values(profile_df[V1_FEATURES])
    if isinstance(sv, list):
        sv = sv[1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]
    return score, np.asarray(sv[0]).flatten(), float(base_value)


def shap_waterfall_rows(shap_vals: np.ndarray, base_value: float, profile_df: pd.DataFrame, n: int = 20) -> list[dict[str, Any]]:
    feats = V1_FEATURES
    vals = profile_df[feats].values.flatten()
    order = np.argsort(-np.abs(shap_vals))[:n]
    rows = []
    for i in order:
        rows.append(
            {
                "feature": feats[i],
                "shap": float(shap_vals[i]),
                "value": float(vals[i]),
            }
        )
    return rows


def tier_migration_text(score_before: float, score_after: float) -> str:
    t1, l1 = score_to_tier_num(score_before), score_to_label(score_before)
    t2, l2 = score_to_tier_num(score_after), score_to_label(score_after)
    diff = score_after - score_before
    arrow = "\u2191" if diff > 0 else "\u2193"
    return (
        f"Tier {t1} ({l1}) \u2192 Tier {t2} ({l2})  |  "
        f"Score: {score_before:.4f} {arrow} {score_after:.4f} "
        f"({'+' if diff > 0 else ''}{diff:.4f})"
    )


def feature_delta_table(
    profile_before: pd.DataFrame,
    profile_after: pd.DataFrame,
    shap_before: np.ndarray,
    shap_after: np.ndarray,
    features: list[str],
    n: int = 20,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(
        {
            "feature": features,
            "original_value": profile_before[features].values.flatten(),
            "modified_value": profile_after[features].values.flatten(),
            "value_change": (profile_after[features].values - profile_before[features].values).flatten(),
            "shap_original": shap_before,
            "shap_modified": shap_after,
            "shap_delta": shap_after - shap_before,
        }
    )
    df["abs_shap_delta"] = df["shap_delta"].abs()
    df = df.sort_values("abs_shap_delta", ascending=False).head(n).reset_index(drop=True)
    return json.loads(df.to_json(orient="records"))


# --- Preset scenarios (unchanged from notebook) ---


def scenario_single_delinquency(profile: pd.DataFrame, amount: float = 200.0) -> pd.DataFrame:
    p = profile.copy()
    for f in [f for f in V1_FEATURES if f.startswith("delinquent_cnt")]:
        p[f] = p[f] + 1
    for f in [f for f in V1_FEATURES if "delinquent_amt" in f and ("sum" in f or "max" in f)]:
        p[f] = p[f] + amount
    for f in [f for f in V1_FEATURES if "delinquent_amt_avg" in f]:
        p[f] = p[f] + amount * 0.5
    for f in [f for f in V1_FEATURES if "insufficient_funds_pct" in f]:
        p[f] = np.minimum(p[f] + 0.15, 1.0)
    for f in [f for f in V1_FEATURES if "insufficient_funds_amt" in f]:
        p[f] = p[f] + amount
    return p


def scenario_cure_delinquencies(profile: pd.DataFrame) -> pd.DataFrame:
    p = profile.copy()
    for f in [f for f in V1_FEATURES if f.startswith("delinquent_")]:
        p[f] = 0.0
    for f in [f for f in V1_FEATURES if f.startswith("ontime_amt_")]:
        p[f] = p[f] * 1.5
    for f in [f for f in V1_FEATURES if f.startswith("paid_amt_sum")]:
        p[f] = p[f] * 1.3
    return p


def scenario_missed_retries(profile: pd.DataFrame, retry_pct_increase: float = 0.3) -> pd.DataFrame:
    p = profile.copy()
    for f in [f for f in V1_FEATURES if f.startswith("retry_charge_pct_")]:
        p[f] = np.minimum(p[f] + retry_pct_increase, 1.0)
    for f in [f for f in V1_FEATURES if f.startswith("charge_approved_pct_")]:
        p[f] = np.maximum(p[f] - retry_pct_increase, 0.0)
    for f in [f for f in V1_FEATURES if f.startswith("retry_charge_amt_")]:
        p[f] = p[f] * 1.5
    return p


def scenario_consistent_ontime(profile: pd.DataFrame, months: int = 3) -> pd.DataFrame:
    p = profile.copy()
    multiplier = 1.0 + 0.15 * months
    for f in [f for f in V1_FEATURES if f.startswith("ontime_amt_")]:
        p[f] = p[f] * multiplier
    for f in [f for f in V1_FEATURES if f.startswith(("paid_amt_sum", "paid_amt_max", "paid_cnt_"))]:
        p[f] = p[f] * multiplier
    for f in [f for f in V1_FEATURES if f.startswith("early_amt_")]:
        p[f] = p[f] * (1.0 + 0.1 * months)
    for f in [f for f in V1_FEATURES if f.startswith(("late_amt_", "late_cnt_"))]:
        p[f] = p[f] * max(0.3, 1.0 - 0.2 * months)
    return p


def scenario_new_card(profile: pd.DataFrame) -> pd.DataFrame:
    p = profile.copy()
    for f in [f for f in V1_FEATURES if f.startswith("card_added_cnt_")]:
        p[f] = p[f] + 1
    for f in [f for f in V1_FEATURES if f.startswith("payment_card_first_use_cnt_")]:
        p[f] = p[f] + 1
    for f in [f for f in V1_FEATURES if f.startswith("payment_card_cnt_")]:
        p[f] = p[f] + 1
    if "card_active_cnt_lag1d" in V1_FEATURES:
        p["card_active_cnt_lag1d"] = p["card_active_cnt_lag1d"] + 1
    return p


PRESET_SCENARIOS: dict[str, dict[str, Any]] = {
    "Single order delinquency": {
        "fn": scenario_single_delinquency,
        "params": {"amount": 200.0},
        "description": "Good customer misses one payment",
    },
    "Cure all delinquencies": {
        "fn": scenario_cure_delinquencies,
        "params": {},
        "description": "Bad customer pays off all delinquent balances",
    },
    "Missed retry charges": {
        "fn": scenario_missed_retries,
        "params": {"retry_pct_increase": 0.3},
        "description": "Payment retries start failing",
    },
    "Consistent on-time payments (3 months)": {
        "fn": scenario_consistent_ontime,
        "params": {"months": 3},
        "description": "Customer pays on time for 3 months",
    },
    "New card added": {
        "fn": scenario_new_card,
        "params": {},
        "description": "Customer adds a fresh payment card",
    },
}
