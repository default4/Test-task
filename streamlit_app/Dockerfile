FROM python:3.9-slim

RUN pip install --upgrade pip

# Скопируем requirements.txt для Streamlit
COPY requirements.txt /tmp/requirements_streamlit.txt
RUN pip install --no-cache-dir -r /tmp/requirements_streamlit.txt

# Определяем рабочую директорию
WORKDIR /app

# Здесь можно либо скопировать всё (app.py) в /app, либо опираться на volume в docker-compose.
COPY . /app

# Запуск по умолчанию
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
