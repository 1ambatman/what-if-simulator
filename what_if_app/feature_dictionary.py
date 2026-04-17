"""Unified RCM V1 feature descriptions from bundled CSV (data dictionary)."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

_DATA_CSV = Path(__file__).resolve().parent / "data" / "unified_rcm_v1_features.csv"


@lru_cache(maxsize=1)
def get_feature_descriptions() -> dict[str, str]:
    """Map feature column name -> human-readable description for tooltips."""
    if not _DATA_CSV.is_file():
        return {}
    out: dict[str, str] = {}
    with _DATA_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = (row.get("Feature") or "").strip()
            desc = (row.get("Feature Description") or "").strip()
            if feat:
                out[feat] = desc
    return out
