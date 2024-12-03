import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


def load_images():
    """Загружает изображения из каталога 'fonts' и преобразует их в одномерные векторы."""
    images = []
    for filename in os.listdir("fonts"):
        img = Image.open(os.path.join("fonts", filename)).convert("L")
        images.append(np.array(img).flatten())
    return np.array(images)


# Шаг 1: Загрузка и линеаризация изображений
image_matrix = load_images()

# Шаг 2: Центрирование данных (вычитание среднего)
mean_image = np.mean(image_matrix, axis=0)
centered_images = image_matrix - mean_image

# Шаг 3: Стандартизация данных
scaler = StandardScaler()
standardized_images = scaler.fit_transform(centered_images)

# Шаг 4: Применение PCA для уменьшения размерности
pca = PCA(n_components=100)  # Выбираем 100 главных компонент
principal_components = pca.fit_transform(standardized_images)


# Шаг 5: Масштабирование PCA-результатов в диапазон [0, 1]
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_principal_components = minmax_scaler.fit_transform(principal_components)

# Приведение к диапазону [0, 255] для визуализации или сохранения
scaled_principal_components_255 = (scaled_principal_components * 255).astype(np.uint8)

# Шаг 6: Визуализация первых двух компонент
plt.scatter(
    scaled_principal_components_255[:, 0],
    scaled_principal_components_255[:, 1],
    alpha=0.7,
)
plt.title("Перші дві головні компоненти (масштабовані в [0, 255])")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.savefig("results.jpg")

# Проверка диапазона значений
