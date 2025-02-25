#!/usr/bin/env bash
set -e

# Используем db upgrade вместо устаревшего db init
airflow db init

echo "Creating admin user..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com || true

# Если вам нужно выполнить пользовательский SQL-скрипт (например, создать таблицу в отдельной схеме),
# можно добавить здесь команду, например:
echo "Loading custom schema..."
export PGPASSWORD=airflow && psql -h postgres -U airflow -d airflow -f /opt/airflow/db/schema.sql

echo "Starting Airflow scheduler and webserver..."
# Запускаем scheduler в фоне, а затем переходим к webserver
airflow scheduler &
exec airflow webserver
