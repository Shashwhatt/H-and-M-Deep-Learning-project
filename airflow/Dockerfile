FROM apache/airflow:2.10.5-python3.11

USER airflow

RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    "pandas>=2.2,<3" \
    "redis>=5,<9"

RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    "tensorflow==2.21.0" \
    "tensorflow-recommenders==0.7.7" \
    "tf-keras==2.21.0"
