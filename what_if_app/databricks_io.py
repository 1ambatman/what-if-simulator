"""Load customer rows from Databricks SQL (Unity Catalog tables)."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from databricks import sql

from what_if_app.config import settings


def _sanitize_table_name(table: str) -> str:
    """Allow only safe Unity Catalog three-part names."""
    t = table.strip()
    if not re.fullmatch(r"[\w.-]+\.[\w.-]+\.[\w.-]+", t):
        raise ValueError("Table must look like catalog.schema.table (letters, numbers, underscores, dots).")
    return t


def _connection():
    if not settings.databricks_host or not settings.databricks_http_path or not settings.databricks_token:
        raise RuntimeError(
            "Databricks is not configured. Set DATABRICKS_HOST, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN."
        )
    return sql.connect(
        server_hostname=settings.databricks_host.replace("https://", "").rstrip("/"),
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_token,
    )


def _pairs_union_sql(pairs: list[tuple[str, str]]) -> str:
    parts = []
    for cid, d in pairs:
        esc = cid.replace("'", "''")
        parts.append(f"SELECT '{esc}' AS customer_id, CAST('{d}' AS DATE) AS reference_date")
    return " UNION ALL ".join(parts)


def fetch_profiles_from_predictions_table(
    table: str,
    pairs: list[tuple[str, str]],
    prediction_info_column: str = "prediction_info",
) -> pd.DataFrame:
    """
    Fetch model_score and prediction_info for each (customer_id, reference_date) pair.
    Parses JSON from prediction_info to avoid hundreds of try_variant_get columns in SQL.
    """
    if not pairs:
        return pd.DataFrame()
    t = _sanitize_table_name(table)
    union_sql = _pairs_union_sql(pairs)
    query = f"""
    WITH pairs AS (
      {union_sql}
    ),
    joined AS (
      SELECT
        p.customer_id,
        CAST(p.prediction_timestamp AS DATE) AS reference_date,
        p.model_score,
        TO_JSON(p.{prediction_info_column}) AS prediction_info_json,
        ROW_NUMBER() OVER (
          PARTITION BY p.customer_id, CAST(p.prediction_timestamp AS DATE)
          ORDER BY p.prediction_timestamp DESC
        ) AS rn
      FROM {t} AS p
      INNER JOIN pairs AS q
        ON p.customer_id = q.customer_id
        AND CAST(p.prediction_timestamp AS DATE) = q.reference_date
    )
    SELECT customer_id, reference_date, model_score, prediction_info_json
    FROM joined
    WHERE rn = 1
    """
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
    return pd.DataFrame(rows, columns=cols)


def fetch_customer_pairs_from_input_table(input_table: str) -> list[tuple[str, str]]:
    t = _sanitize_table_name(input_table)
    query = f"""
    SELECT DISTINCT
      CAST(customer_id AS STRING) AS customer_id,
      CAST(reference_date AS STRING) AS reference_date
    FROM {t}
    """
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return [(str(r[0]), str(r[1])[:10]) for r in rows]


def parse_prediction_json(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        obj = raw
    else:
        try:
            obj = json.loads(raw)
            if isinstance(obj, str):
                obj = json.loads(obj)
        except (TypeError, json.JSONDecodeError):
            return {}
    if "input" in obj and isinstance(obj["input"], dict):
        return obj["input"]
    return obj if isinstance(obj, dict) else {}
