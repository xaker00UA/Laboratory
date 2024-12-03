import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing,
)


# Функція для завантаження зображення та перетворення в відтінки сірого
def load_grayscale(image_path):
    img = Image.open(image_path).convert("L")  # Перетворення в відтінки сірого
    return np.array(img)


# Функція для порогового перетворення (бінаризації)
def threshold_image(img, threshold=128):
    binary_img = (img > threshold).astype(np.uint8)  # Порогове значення
    return binary_img


# Функція для створення структурного елемента у вигляді послідовності точок
def create_structuring_element(size):
    element = np.zeros((size, size), dtype=np.uint8)
    np.fill_diagonal(element, 1)  # Заповнення діагоналі
    return element


# Функція для відображення зображення
def show_image(img, title=""):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"images/{title}.jpg")
    plt.close()


# Основний блок обробки зображення
def main():
    image_path = "stones_24.jpg"
    # 1. Завантаження та перетворення в відтінки сірого
    grayscale_img = load_grayscale(image_path)
    show_image(grayscale_img, "Відтінки сірого")

    # 2. Порогове перетворення
    binary_img = threshold_image(grayscale_img, threshold=128)
    show_image(binary_img, "Бінарне зображення")

    # 3. Ерозія зображення структурним елементом (послідовність точок)
    struct_element = create_structuring_element(5)  # Послідовність точок розміром 5x5
    eroded_img = binary_erosion(binary_img, structure=struct_element).astype(np.uint8)
    show_image(eroded_img, "Ерозія (послідовність точок)")

    # 4. Дилатація зображення
    dilated_img = binary_dilation(eroded_img, structure=struct_element).astype(np.uint8)
    show_image(dilated_img, "Дилатація")

    # 5. Морфологічне розмикання
    opened_img = binary_opening(binary_img, structure=struct_element).astype(np.uint8)
    show_image(opened_img, "Морфологічне розмикання")

    # 6. Морфологічне замикання
    closed_img = binary_closing(opened_img, structure=struct_element).astype(np.uint8)
    show_image(closed_img, "Морфологічне замикання")


if __name__ == "__main__":
    main()
