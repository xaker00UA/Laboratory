import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_images(image_folder, target_size=(100, 100)):
    """Загружает изображения из папки и преобразует их в одномерные векторы, приводя их к одному размеру."""
    images = []
    for filename in os.listdir(image_folder):
        img = Image.open(os.path.join(image_folder, filename)).convert("L")
        img = img.resize(target_size)  # Приводим изображения к нужному размеру
        images.append(np.array(img).flatten())
    return np.array(images)


def plot_images(images, title, num_images=10, image_size=(100, 100)):
    """Отображает первые num_images изображений из класса."""
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(image_size), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


def pca_kmeans_cluster(
    image_matrix, n_components_list, n_clusters=3, image_size=(100, 100)
):
    """Применяет PCA и кластеризацию K-средних для разных чисел главных компонент."""
    for n_components in n_components_list:
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(image_matrix)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(pca_components)

        labels = kmeans.labels_

        for label in range(n_clusters):
            class_images = image_matrix[labels == label]
            plot_images(
                class_images,
                f"K-means with {n_components} components - Class {label + 1}",
                image_size=image_size,
            )


def cluster_image_pixels(image_path, n_clusters_list, patch_size=(100, 100)):
    """Кластеризация пикселей изображения для разных количеств кластеров."""
    img = Image.open(image_path)
    img = img.convert("RGB")
    img_array = np.array(img)

    # Вычисляем количество патчей по вертикали и горизонтали
    patches_vertical = img_array.shape[0] // patch_size[0]
    patches_horizontal = img_array.shape[1] // patch_size[1]
    total_patches = patches_vertical * patches_horizontal

    for n_clusters in n_clusters_list:
        # Если количество патчей меньше числа кластеров, пропускаем кластеризацию
        if total_patches < n_clusters:
            print(
                f"Для {n_clusters} кластеров недостаточно патчей ({total_patches}). Пропускаем."
            )
            continue

        patches = []

        # Разбиение изображения на патчи (части)
        for i in range(0, img_array.shape[0], patch_size[0]):
            for j in range(0, img_array.shape[1], patch_size[1]):
                patch = img_array[i : i + patch_size[0], j : j + patch_size[1]]
                patches.append(patch.flatten())
        patches = np.array(patches)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(patches)

        clustered_img = kmeans.labels_.reshape(img_array.shape[0], img_array.shape[1])

        plt.imshow(clustered_img, cmap="viridis")
        plt.title(f"K-Means with {n_clusters} clusters")
        plt.show()


def spectral_clustering_with_ds(image_matrix, n_clusters=3):
    """Применяет спектральную кластеризацию с использованием матрицы D^-1 S."""
    dist_matrix = pairwise_distances(image_matrix, metric="euclidean")

    # Создаем матрицу A = D^-1 S
    row_sums = dist_matrix.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sums))
    A = np.dot(np.dot(D_inv_sqrt, dist_matrix), D_inv_sqrt)

    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=42
    )
    labels = spectral.fit_predict(A)

    for label in range(n_clusters):
        class_images = image_matrix[labels == label]
        plot_images(
            class_images,
            f"Spectral Clustering with {n_clusters} clusters - Class {label + 1}",
        )


if __name__ == "__main__":
    # Загрузка изображений
    image_folder = "fonts"
    image_matrix = load_images(image_folder, target_size=(100, 100))

    # # Применение PCA и кластеризация с использованием K-средних для различных компонент
    # pca_kmeans_cluster(image_matrix, n_components_list=[5, 10, 20], n_clusters=3)

    # Кластеризация пикселей изображения
    image_path = "fonts/2201_t.jpg"  # Укажите путь к своему изображению
    cluster_image_pixels(image_path, n_clusters_list=[2, 3, 5], patch_size=(100, 100))

    # Применение спектральной кластеризации с матрицей D^-1 S
    spectral_clustering_with_ds(image_matrix, n_clusters=3)
