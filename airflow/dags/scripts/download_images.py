import os
import shutil

def download_dataset(**kwargs):
    """
    Функция download_dataset предназначена для копирования изображений из локальной директории,
    где они хранятся на хост-машине, в директорию внутри контейнера Airflow
    
    Пример исходного пути (на хост-машине): 
      "C:\\Users\\Professional\\Desktop\\data\\images"
      
    Функция сначала пытается получить путь из переменной окружения LOCAL_IMAGES_PATH
    Если переменная не установлена, используется путь по умолчанию ("/opt/airflow/local_images")
    
    Все найденные изображения (с расширениями .jpg, .jpeg, .png) копируются в директорию
      /opt/airflow/images/raw
    внутри контейнера, создавая её, если она ещё не существует
    """

    # Получаем путь к исходной директории с изображениями из переменной окружения
    source_dir = os.environ.get("LOCAL_IMAGES_PATH", "/opt/airflow/local_images")
    
    # Определяем директорию назначения внутри контейнера, куда будут копироваться изображения
    dest_dir = "/opt/airflow/images/raw"

    # Создаем директорию назначения, если она еще не существует
    os.makedirs(dest_dir, exist_ok=True)

    # Определяем допустимые расширения файлов для копирования
    allowed_extensions = {".jpg", ".jpeg", ".png"}

    # Перебираем все файлы в исходной директории
    for file in os.listdir(source_dir):
        # Получаем расширение файла и приводим его к нижнему регистру для корректного сравнения
        file_ext = os.path.splitext(file)[1].lower()
        # Если расширение файла входит в набор разрешенных, копируем файл
        if file_ext in allowed_extensions:
            src_file = os.path.join(source_dir, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

    # Функция возвращает True, сигнализируя об успешном завершении операции
    return True
