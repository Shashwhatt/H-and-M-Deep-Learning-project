
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="hm_daily_embedding_update",
    description="Refresh article embeddings in Redis every day",
    schedule="0 3 * * *",
    start_date=pendulum.datetime(
        2026, 11, 9, tz="Asia/Kolkata"
    ),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "hm-recommendation-team",
        "retries": 1,
    },
    tags=["hm", "redis", "embeddings"],
) as dag:

    refresh_embeddings = BashOperator(
        task_id="generate_and_store_article_embeddings",
        bash_command="python -m src.update_redis_embeddings",
        cwd=PROJECT_DIR,
        env={
            "TF_USE_LEGACY_KERAS": "1",
            "PYTHONPATH": PROJECT_DIR,
            "REDIS_HOST": "host.docker.internal",
            "REDIS_PORT": "6379",
        },
        append_env=True,
    )
