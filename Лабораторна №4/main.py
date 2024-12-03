import os
from PIL import Image, ImageFont, ImageDraw
from numpy import array, dot, linalg, sqrt, mean
import matplotlib.pyplot as plt


# Функція для завантаження зображень із папки fonts
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("L")  # Перетворення в градації сірого
        images.append(array(img).flatten())  # Перетворення в 1D-вектор
    return array(images), img.size


# PCA: Аналіз головних компонент
def pca(X):
    num_data, dim = X.shape
    mean_X = mean(X, axis=0)
    X = X - mean_X
    if dim > num_data:
        M = dot(X, X.T)
        e, EV = linalg.eigh(M)
        tmp = dot(X.T, EV).T
        V = tmp[::-1]
        S = sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        U, S, V = linalg.svd(X)
        V = V[:num_data]
    return V, S, mean_X


# Візуалізація головних компонент
def plot_pca_components(components, img_size):
    for i in range(len(components)):
        plt.imshow(components[i].reshape(img_size), cmap="gray")
        plt.title(f"Головна компонента {i + 1}")
        plt.axis("off")
        plt.savefig(f"images/Головна компонента {i + 1}.jpg")
        plt.close()


# Основна функція
def main():
    # Завантаження зображень
    images, img_size = load_images_from_folder("fonts")

    # Виконання PCA
    components, _, _ = pca(images)

    # Візуалізація перших трьох головних компонент
    plot_pca_components(components[:3], img_size)


if __name__ == "__main__":
    main()
