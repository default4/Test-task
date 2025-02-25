import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
import json

def train_model(**kwargs):
    """
    Функция train_model выполняет следующие задачи:
    
    1. Извлекает данные предварительной обработки изображений из XCom, 
       декодирует их из JSON, если необходимо, и извлекает список обработанных записей 
       
    2. Отбирает валидные изображения (где is_valid=True) для дальнейшей работы
    
    3. Случайным образом перемешивает валидные изображения и делит их на обучающую (train) 
       и тестовую (test) выборки для обучения CNN-модели
    
    4. Определяет функцию get_label для определения метки ("cat" или "dog") на основе имени файла
    
    5. С помощью функции load_data загружает изображения и соответствующие метки, 
       преобразуя изображения в массивы NumPy с нормализацией (значения от 0 до 1)
    
    6. Если данных достаточно, начинается MLflow эксперимент:
       - Логируются параметры (например, размеры обучающей и тестовой выборок, число эпох, форма входных данных)
       - Создаётся простая CNN-модель для классификации изображений (2 класса: cat и dog)
       - Модель обучается в течение 2 эпох с валидационным разделением
       - Логируются метрики (потери и точность) для обучающей и валидационной выборок по каждой эпохе
       - Модель оценивается на тестовой выборке, и результаты также логируются
       - Модель сохраняется (логируется) с помощью mlflow.tensorflow.log_model
    
    7. После обучения модель применяется ко всем валидным изображениям для получения предсказаний
       Каждому изображению присваивается метка "cat" или "dog" на основе результата работы модели
    
    8. Функция возвращает исходный результат предварительной обработки (с обновлёнными метками)
    """

    # Извлекаем данные из XCom, полученные предыдущей задачей preprocess_images
    ti = kwargs["ti"]
    preprocessed_info = ti.xcom_pull(task_ids="preprocess_images", key="return_value")
    
    # Если данные получены в виде строки (например, JSON-формат), декодируем их
    if isinstance(preprocessed_info, str):
        preprocessed_info = json.loads(preprocessed_info)
    
    # Если данные представляют собой словарь и содержат ключ "results",
    # извлекаем список обработанных изображений и время обработки
    if isinstance(preprocessed_info, dict) and "results" in preprocessed_info:
        items = preprocessed_info["results"]
        processing_time = preprocessed_info.get("processing_time", 0)
    else:
        items = preprocessed_info
        processing_time = 0

    # Фильтруем валидные изображения (где is_valid=True)
    valid_items = [x for x in items if x.get("is_valid")]

    # Перемешиваем список валидных изображений случайным образом для случайного разделения на train/test
    random.shuffle(valid_items)
    # Берем первые 80 изображений для обучения, следующие 20 для тестирования
    train_items = valid_items[:80]
    test_items = valid_items[80:100]

    # Функция для определения метки изображения по имени файла
    def get_label(filename):
        fn = filename.lower()
        if "cat" in fn:
            return 0
        elif "dog" in fn:
            return 1
        return None

    # Функция load_data загружает изображения и соответствующие метки,
    # преобразует изображения в массивы NumPy и нормализует значения
    def load_data(items):
        X, y = [], []
        for it in items:
            label = get_label(it.get("filename"))
            if label is not None:
                with Image.open(it.get("processed_path")) as img:
                    # Преобразуем изображение в массив NumPy; ожидается размер 256x256
                    arr = np.array(img)  # Получаем двумерный массив
                    # Добавляем размерность для канала (например, grayscale -> (256,256,1))
                    arr = np.expand_dims(arr, axis=-1)
                    X.append(arr)
                    y.append(label)
        # Приводим список к массиву float32 и нормализуем (значения от 0 до 1)
        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.int32)
        return X, y

    # Загружаем данные для обучающей и тестовой выборок
    X_train, y_train = load_data(train_items)
    X_test, y_test = load_data(test_items)

    # Настраиваем эксперимент MLflow
    mlflow.set_experiment("Cats vs Dogs CNN")
    with mlflow.start_run(run_name="Simple CNN training"):
        # Логируем параметры эксперимента
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("epochs", 2)
        mlflow.log_param("input_shape", str(X_train.shape[1:]))
        
        # Определяем простую сверточную нейронную сеть (CNN)
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(2, activation='softmax')  # Выходной слой для 2 классов: cat и dog
        ])

        # Компилируем модель с оптимизатором Adam и функцией потерь для многоклассовой классификации
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Обучаем модель на обучающих данных с валидационным разделением (20%)
        history = model.fit(X_train, y_train, epochs=2, validation_split=0.2, verbose=1)

        # Логируем метрики для каждой эпохи
        for epoch_idx, (loss_val, acc_val) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
            mlflow.log_metric("train_loss", loss_val, step=epoch_idx)
            mlflow.log_metric("train_accuracy", acc_val, step=epoch_idx)
        for epoch_idx, (val_loss_val, val_acc_val) in enumerate(zip(history.history['val_loss'], history.history['val_accuracy'])):
            mlflow.log_metric("val_loss", val_loss_val, step=epoch_idx)
            mlflow.log_metric("val_accuracy", val_acc_val, step=epoch_idx)

        # Оцениваем модель на тестовой выборке и логируем метрики
        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", acc)

        # Сохраняем модель в артефакты MLflow
        mlflow.tensorflow.log_model(model, artifact_path="model")

        # Функция для предсказания метки изображения на основе пути к обработанному файлу
        def predict_label(path):
            with Image.open(path) as img:
                arr = np.array(img)
                arr = np.expand_dims(arr, axis=-1)  # Добавляем канал, если изображение grayscale
                arr = np.expand_dims(arr, axis=0)   # Добавляем размерность батча
                arr = arr.astype(np.float32) / 255.0
            pred = model.predict(arr)
            label_idx = np.argmax(pred, axis=1)[0]
            return "cat" if label_idx == 0 else "dog"

        # Для каждого валидного изображения делаем предсказание и сохраняем результат в поле "label"
        for item in valid_items:
            item["label"] = predict_label(item.get("processed_path"))

    # Возвращаем обновленные данные предварительной обработки (с добавленными метками)
    return preprocessed_info
