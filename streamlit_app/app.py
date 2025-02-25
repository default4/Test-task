import streamlit as st
import psycopg2
import pandas as pd
import altair as alt

def get_db_connection():
    """
    Функция для установления соединения с базой данных PostgreSQL
    Используются следующие параметры подключения:
      - dbname: имя базы данных ("airflow")
      - user: имя пользователя ("airflow")
      - password: пароль ("airflow")
      - host: хост (имя сервиса "postgres" в docker-compose)
      - port: порт (5432)
    Возвращается объект соединения
    """
    return psycopg2.connect(
        dbname="airflow",
        user="airflow",
        password="airflow",
        host="postgres",
        port=5432
    )

def main():
    # Заголовок приложения
    st.title("Cats vs Dogs - Image Processing Dashboard")

    # Устанавливаем соединение с базой данных и считываем данные из таблицы image_metadata
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM image_metadata ORDER BY id DESC", conn)
    conn.close()

    # Выводим на экран последние 50 записей из таблицы для предварительного просмотра
    st.subheader("Последние записи из image_metadata")
    st.write(df.head(50))

    # ---------------------------------------------------------------------------------------
    # 1) Определяем метку (label) для каждого изображения на основе начала строки в поле filename
    # Если имя файла начинается с "cat", изображение считается кошкой
    # Если начинается с "dog", изображение считается собакой
    # Если не начинается ни с одного из указанных, присваивается метка "unknown"
    # ---------------------------------------------------------------------------------------
    def detect_label(filename):
        if filename.lower().startswith("cat"):
            return "cat"
        elif filename.lower().startswith("dog"):
            return "dog"
        return "unknown"

    # Применяем функцию detect_label к столбцу "filename" и сохраняем результат в новой колонке "detected_label"
    df["detected_label"] = df["filename"].apply(detect_label)

    # ---------------------------------------------------------------------------------------
    # 2) Визуализируем соотношение кошек и собак.
    # Используем value_counts для подсчёта количества записей для каждого класса и строим bar chart
    # ---------------------------------------------------------------------------------------
    st.subheader("Соотношение (Cat vs Dog)")
    label_counts = df["detected_label"].value_counts()
    st.bar_chart(label_counts)

    # ---------------------------------------------------------------------------------------
    # Построение круговой диаграммы (pie chart) с помощью Altair,
    # которая наглядно показывает доли каждого класса
    # ---------------------------------------------------------------------------------------
    catdog_chart = alt.Chart(
        df["detected_label"].value_counts().rename_axis("label").reset_index(name="count")
    ).mark_arc(innerRadius=50).encode(
        theta="count:Q",      # Угол сектора пропорционален количеству
        color="label:N",      # Цвет сектора зависит от метки
        tooltip=["label:N", "count:Q"]  # При наведении показываются метка и количество
    ).properties(
        width=400,
        height=400
    )
    st.altair_chart(catdog_chart, use_container_width=True)

    # ---------------------------------------------------------------------------------------
    # 3) Построение гистограммы распределения размеров файлов
    # Используем Altair для построения гистограммы, где по оси X отложен размер файла в байтах,
    # а по оси Y – количество изображений, попавших в соответствующий диапазон
    # ---------------------------------------------------------------------------------------
    st.subheader("Распределение размера файлов (file_size)")
    file_size_chart = alt.Chart(df).mark_bar().encode(
        alt.X("file_size:Q", bin=alt.Bin(maxbins=30), title="File size (bytes)"),
        alt.Y("count()", title="Count of images"),
        tooltip=["count()"]
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(file_size_chart, use_container_width=True)

if __name__ == "__main__":
    main()
