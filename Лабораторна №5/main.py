import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.spatial.distance import cdist


# Завантаження зображень
def load_images(folder):
    images = []
    for file in os.listdir(folder):
        img = Image.open(os.path.join(folder, file)).convert("L")
        images.append(np.array(img).flatten())
    return np.array(images)


# Функція для кластеризації методом К-середніх
def kmeans(X, k, max_iter=100):
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iter):
        distances = cdist(X, centroids, "euclidean")
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids


# # Візуалізація зображень кластеру
# def visualize_clusters(
#     images, labels, cluster_num, img_size=(25, 25), save_folder="images"
# ):
#     cluster_images = images[labels == cluster_num][:10]  # 10 зображень
#     for idx, img in enumerate(cluster_images):
#         # Створюємо шлях для збереження
#         save_path = os.path.join(
#             save_folder, f"cluster_{cluster_num}_image_{idx+1}.png"
#         )
#         plt.imshow(img.reshape(img_size), cmap="gray")
#         plt.axis("off")
#         plt.savefig(
#             save_path, bbox_inches="tight", pad_inches=0
#         )  # Зберігаємо зображення
#         plt.close()  # Закриваємо поточну фігуру, щоб уникнути перекриття
def visualize_clusters(
    images, labels, cluster_num, img_size=(25, 25), save_folder="images"
):
    cluster_images = images[labels == cluster_num][:10]  # 10 зображень
    for idx, img in enumerate(cluster_images):
        # Переводим изображение в правильный формат и изменяем его размер
        img = Image.fromarray(img.reshape(img_size))  # Преобразуем в формат PIL
        img = img.resize(img_size)  # Изменяем размер изображения

        # Создаем путь для сохранения
        save_path = os.path.join(
            save_folder, f"cluster_{cluster_num}_image_{idx+1}.png"
        )

        # Сохраняем изображение
        img.save(save_path)


# Основна функція
def main():
    images = load_images("fonts")
    for n_components in [5, 10, 20]:
        # PCA вручну
        mean = np.mean(images, axis=0)
        centered_data = images - mean
        U, S, V = np.linalg.svd(centered_data, full_matrices=False)
        pca_data = np.dot(centered_data, V[:n_components].T)

        # К-середніх
        labels, _ = kmeans(pca_data, 3)
        for i in range(3):
            print(f"Кластер {i+1} для {n_components} компонент:")
            visualize_clusters(images, labels, i)


if __name__ == "__main__":
    main()
