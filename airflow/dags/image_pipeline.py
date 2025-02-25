import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from minio import Minio
import mlflow
import json

# Импорт логики для обработки изображений и работы с ML-моделью из папки scripts
# Эти модули должны быть доступны в volume, см. настройки docker-compose
from scripts import download_images, preprocess_images, data_validation, ml_classification

# Задаём параметры для подключения к MinIO. Они могут быть переопределены через переменные окружения
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "processed-images"

# Настройка базовых параметров для DAG Airflow
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Создаём DAG с именем "image_pipeline". Здесь описывается весь ETL-пайплайн для обработки изображений,
# включая загрузку, валидацию, предобработку, загрузку в MinIO, сохранение метаданных в PostgreSQL,
# логирование в MLflow и обучение модели
dag = DAG(
    "image_pipeline",
    default_args=default_args,
    description="ETL pipeline for images with ClearML logging",
    schedule_interval="@daily",
    catchup=False,
)

###########################################################################
# Функция upload_to_minio
#
# Эта функция извлекает результаты обработки изображений (например, после задачи preprocess_images),
# и затем загружает обработанные изображения в S3-совместимый MinIO bucket
# Если данные из XCom передаются в виде словаря, извлекается ключ "results"
###########################################################################
def upload_to_minio(**kwargs):
    from minio import Minio
    import os

    ti = kwargs["ti"]
    # Извлекаем данные, возвращённые предыдущей задачей, через XCom
    data = ti.xcom_pull(task_ids="preprocess_images", key="return_value")
    
    # Если данные представлены в виде словаря, извлекаем список результатов из ключа "results"
    if isinstance(data, dict):
        processed_data = data.get("results", [])
    else:
        processed_data = data

    # Инициализируем клиента MinIO с параметрами, полученными из переменных окружения
    client = Minio(
        endpoint=os.environ.get("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=False
    )
    
    MINIO_BUCKET = "processed-images"
    # Если bucket не существует, создаём его
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    
    # Для каждого обработанного изображения копируем файл в указанный bucket
    for item in processed_data:
        # Предполагается, что item — это словарь с информацией об изображении
        filename = item["filename"]
        path = item["processed_path"]
        with open(path, "rb") as f:
            file_stat = os.stat(path)
            client.put_object(
                bucket_name=MINIO_BUCKET,
                object_name=filename,
                data=f,
                length=file_stat.st_size,
                content_type="image/jpeg"
            )

###########################################################################
# Функция store_metadata
#
# Эта функция получает результаты обработки изображений (через XCom) и сохраняет их
# в таблице image_metadata в базе данных PostgreSQL
###########################################################################
def store_metadata(**kwargs):
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    ti = kwargs["ti"]
    data = ti.xcom_pull(task_ids="preprocess_images", key="return_value")
    
    # Если данные представлены в виде словаря, извлекаем список результатов из ключа "results"
    if isinstance(data, dict):
        processed_data = data.get("results", [])
    else:
        processed_data = data

    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    sql = """
        INSERT INTO image_metadata (filename, width, height, format, file_size)
        VALUES (%s, %s, %s, %s, %s);
    """
    
    # Для каждого элемента (изображения) выполняется SQL-запрос для сохранения метаданных
    for item in processed_data:
        filename = item["filename"]
        width = item["width"]
        height = item["height"]
        format_ = item["format"]
        file_size = item["file_size"]
        
        pg_hook.run(sql, parameters=(filename, width, height, format_, file_size))

###########################################################################
# Функция create_bucket_if_not_exists
#
# Функция использует boto3 для подключения к S3-совместимому хранилищу (MinIO)
# и проверяет, существует ли bucket с артефактами для MLflow ("mlflow-artifacts")
# Если bucket отсутствует, он создаётся
###########################################################################
def create_bucket_if_not_exists(**kwargs):
    import boto3
    from botocore.exceptions import ClientError

    bucket_name = "mlflow-artifacts"
    endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created.")

###########################################################################
# Функция log_to_mlflow
#
# Эта функция инициирует MLflow эксперимент для логирования данных о процессе обработки изображений
# Она извлекает результаты обработки (и время обработки), логирует параметры и метрики,
# а также сохраняет примеры обработанных изображений как артефакты
###########################################################################
def log_to_mlflow(**kwargs):
    # Начинаем эксперимент MLflow с именем "ImageProcessing"
    mlflow.set_experiment("ImageProcessing")
    with mlflow.start_run(run_name="Daily Pipeline Run") as run:
        ti = kwargs["ti"]
        data = ti.xcom_pull(task_ids="preprocess_images", key="return_value")
        # Если данные представлены как словарь, извлекаем список обработанных изображений и время обработки
        processed_data = data.get("results", []) if isinstance(data, dict) else data
        processing_time = data.get("processing_time", 0) if isinstance(data, dict) else 0
        
        mlflow.log_param("num_images", len(processed_data))
        mlflow.log_metric("processing_time_sec", processing_time)
        
        # Логируем первые 3 обработанных изображения как артефакты
        for i, item in enumerate(processed_data[:3]):
            image_path = item.get("processed_path")
            if image_path:
                mlflow.log_artifact(image_path, artifact_path=f"sample_image_{i+1}")

###########################################################################
# Определение DAG
#
# Ниже описана последовательность выполнения задач:
# 1. download_images – копирование изображений из локальной директории в контейнер
# 2. validate_images – проверка целостности и корректности изображений
# 3. preprocess_images – изменение размера, конвертация и сохранение обработанных изображений
# 4. upload_to_minio – загрузка обработанных изображений в MinIO
# 5. store_metadata – сохранение метаданных изображений в PostgreSQL
# 6. create_bucket – создание bucket для MLflow артефактов, если его нет
# 7. log_to_mlflow – логирование параметров и примеров изображений в MLflow
# 8. train_model – обучение или инференс модели на обработанных данных
###########################################################################
with dag:
    t1 = PythonOperator(
        task_id="download_images",
        python_callable=download_images.download_dataset  # Функция для копирования изображений.
    )

    t2 = PythonOperator(
        task_id="validate_images",
        python_callable=data_validation.validate_images  # Проверка корректности изображений.
    )

    t3 = PythonOperator(
        task_id="preprocess_images",
        python_callable=preprocess_images.preprocess_images,  # Предобработка изображений.
    )

    t4 = PythonOperator(
        task_id="upload_to_minio",
        python_callable=upload_to_minio,  # Загрузка обработанных изображений в MinIO.
    )

    t5 = PythonOperator(
        task_id="store_metadata",
        python_callable=store_metadata,  # Сохранение метаданных изображений в базе данных.
    )

    # Задача для создания bucket в MinIO для хранения артефактов MLflow.
    t6 = PythonOperator(
        task_id="create_bucket",
        python_callable=create_bucket_if_not_exists,
        dag=dag
    )

    t7 = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow,  # Логирование параметров и артефактов в MLflow.
    )

    t8 = PythonOperator(
        task_id="trainmodel",
        python_callable=ml_classification.train_model,  # Обучение модели.
        dag=dag
    )

    # Определяем порядок выполнения задач в DAG.
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8
