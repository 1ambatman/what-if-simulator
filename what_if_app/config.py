from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    databricks_host: str = ""
    databricks_http_path: str = ""
    databricks_token: str = ""

    mlflow_tracking_uri: str = "databricks"
    mlflow_registry_uri: str = "databricks-uc"

    predictions_table: str = "mle.batch_model_inference.predictions"
    mlflow_run_id: str = "9d740e9e5f544d9490100cef238bf074"
    local_model_path: str | None = None

    app_host: str = "127.0.0.1"
    app_port: int = 8765


settings = Settings()
