FROM apache/airflow:2.6.1-python3.9

# Переключаемся на root для установки системных пакетов и настройки entrypoint
USER root

# Копируем entrypoint.sh и меняем его разрешения, пока мы root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Теперь переключаемся на пользователя airflow
USER airflow

# Устанавливаем Python зависимости
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/airflow

ENTRYPOINT ["/entrypoint.sh"]
