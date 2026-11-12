
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project"

COMMON_ENV = {
    "TF_USE_LEGACY_KERAS": "1",
    "PYTHONPATH": PROJECT_DIR,
    "REDIS_HOST": "host.docker.internal",
    "REDIS_PORT": "6379",
}

with DAG(
    dag_id="hm_weekly_model_retraining",
    description="Weekly Two-Tower model retraining and deployment",
    schedule="0 2 * * 1",
    start_date=pendulum.datetime(
        2026, 11, 9, tz="Asia/Kolkata"
    ),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "hm-recommendation-team",
        "retries": 1,
    },
    tags=["hm", "training", "recommendation"],
) as dag:

    train_model = BashOperator(
        task_id="train_two_tower_model",
        bash_command="python src/train_model.py",
        cwd=PROJECT_DIR,
        env={
            **COMMON_ENV,
            "MODEL_OUTPUT_DIR": "models/staging",
        },
        append_env=True,
    )

    validate_and_promote = BashOperator(
        task_id="validate_and_promote_model",
        bash_command="python -m src.promote_model",
        cwd=PROJECT_DIR,
        env=COMMON_ENV,
        append_env=True,
    )

    update_embeddings = BashOperator(
        task_id="update_redis_article_embeddings",
        bash_command="python -m src.update_redis_embeddings",
        cwd=PROJECT_DIR,
        env=COMMON_ENV,
        append_env=True,
    )

    train_model >> validate_and_promote >> update_embeddings
