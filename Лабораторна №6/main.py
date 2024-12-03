import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing,
)
from scipy.ndimage import label
import os


# Завантаження зображення
def load_image(image_path):
    img = Image.open(image_path).convert("L")  # Перетворення в градації сірого
    img = np.array(img)
    return img


# Бінаризація зображення за допомогою порогового значення
def binarize_image(img, threshold=128):
    return img > threshold


# Операція дилатації (розширення)
def dilation(A, B):
    return binary_dilation(A, structure=B)


# Операція ерозії (стиснення)
def erosion(A, B):
    return binary_erosion(A, structure=B)


# Операція відкриття (відмикання)
def opening(A, B):
    return binary_opening(A, structure=B)


# Операція закриття (замикання)
def closing(A, B):
    return binary_closing(A, structure=B)


# Сегментація зображення за вододілами
def watershed_segmentation(A):
    # Лейблінг зображення (пошук окремих об'єктів)
    labeled, num_labels = label(A)
    return labeled


# Збереження результату
def save_image(image, filename):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# Основна функція для виконання морфологічних операцій та сегментації
def main(image_path, struct_elem_shape="square", output_dir="images"):
    # Створення каталогу для збереження результатів, якщо його ще немає
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Завантажуємо та бінаризуємо зображення
    img = load_image(image_path)
    bin_img = binarize_image(img)

    # Створюємо структурний елемент (наприклад, квадрат)
    if struct_elem_shape == "square":
        struct_elem = np.ones((3, 3))  # Квадрат 3x3
    elif struct_elem_shape == "disk":
        struct_elem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # Диск
    else:
        struct_elem = np.ones((3, 3))  # За замовчуванням квадрат

    # Виконуємо морфологічні операції
    dilated = dilation(bin_img, struct_elem)
    eroded = erosion(bin_img, struct_elem)
    opened = opening(bin_img, struct_elem)
    closed = closing(bin_img, struct_elem)

    # Сегментація за вододілами
    watershed_result = watershed_segmentation(bin_img)

    # Візуалізація та збереження результатів
    save_image(dilated, os.path.join("images", f"{image_path[6:-4]}_dilated_image.jpg"))

    save_image(eroded, os.path.join(output_dir, f"{image_path[6:-4]}_eroded_image.jpg"))
    save_image(opened, os.path.join(output_dir, f"{image_path[6:-4]}_opened_image.jpg"))
    save_image(closed, os.path.join(output_dir, f"{image_path[6:-4]}_closed_image.jpg"))
    save_image(watershed_result, os.path.join(output_dir, "watershed_result.jpg"))

    # Виведення результатів


if __name__ == "__main__":
    for image_path in os.listdir("input"):
        main(f"input/{image_path}")
    print("Результати збережено в папку")
