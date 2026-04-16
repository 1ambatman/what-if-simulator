# What-If Score Simulator

Local web app that reproduces the **what_if_simulator** notebook: load customer feature rows from a Databricks Unity Catalog predictions table, score with a **LightGBM** model from **MLflow**, and explore **SHAP**-driven what-if scenarios.

## Prerequisites

- Python 3.10+
- Network access to your Databricks workspace (SQL warehouse + MLflow)
- A personal access token with permission to run SQL on the warehouse and read the MLflow run artifact

## Setup

```bash
cd what-if-simulator
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

If you see `No module named '_cffi_backend'` when the app loads the model, reinstall native deps:

```bash
pip install --force-reinstall 'cffi>=1.16.0' cryptography
```

Edit `.env`:

- **DATABRICKS_HOST** — workspace URL, e.g. `https://adb-xxxx.azuredatabricks.net`
- **DATABRICKS_HTTP_PATH** — SQL warehouse HTTP path, e.g. `/sql/1.0/warehouses/xxxxxxxx`
- **DATABRICKS_TOKEN** — personal access token
- **MLFLOW_TRACKING_URI** — usually `databricks` (uses Databricks-hosted MLflow; ensure your CLI/profile or token is configured per [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html))
- **PREDICTIONS_TABLE** / **MLFLOW_RUN_ID** — defaults match the notebook; override as needed
- **LOCAL_MODEL_PATH** (optional) — path to an exported MLflow LightGBM model directory to skip loading from a run

## Run

```bash
python run.py
```

The app starts at `http://127.0.0.1:8765` and opens your default browser.

Alternatively:

```bash
what-if-simulator
```

## Usage

1. **Load profiles** — Enter the predictions table name, then either paste **customer IDs** and **reference dates** (inline), or point at an **input table** with `customer_id` and `reference_date`.
2. Choose a **profile** and a **scenario** (or **Manual adjustment** to tweak features with sliders).
3. **Run what-if** to see tier migration, score comparison, SHAP drivers, and top feature deltas.

## Data expectations

- Predictions table must include `customer_id`, `prediction_timestamp` (or castable to date), `model_score`, and `prediction_info` (VARIANT/JSON with an `input` object keyed by model feature names), consistent with the original notebook.

If `TO_JSON(prediction_info)` fails in your warehouse, adjust the SQL in `what_if_app/databricks_io.py` (e.g. cast rules for your table).

## License

Use and modify per your organization’s policies. This repository is generated as a standalone tool and is not coupled to the parent monorepo.
