from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing,
    measurements,
)
import inspect

# Завантаження зображення
image_path = "stones_24.jpg"  # Змініть на шлях до вашого зображення
image = Image.open(image_path)


# 1. Відображення оригінального зображення
def show_image(title, img, cmap="gray"):
    func = inspect.stack()[1].function
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{func}.jpg")


def one_task():
    # Перетворення в відтінки сірого
    gray_image = image.convert("L")
    # Бінаризація з використанням порогу (поріг = 128 для стандартного діапазону)
    binary_image = gray_image.point(lambda p: 255 if p > 128 else 0)
    show_image("Відтінки сірого", gray_image)
    show_image("Бінарне зображення", binary_image)
    return binary_image


def two_task():
    # Порогове перетворення (бінаризація)
    threshold = 50  # Поріг можна налаштувати
    gray_image = one_task()
    binary_image = gray_image.point(lambda p: 255 if p > threshold else 0)
    show_image("Бінарне зображення після порогового перетворення", binary_image)
    return np.array(binary_image)  # Повертаємо як масив numpy для подальших операцій


def three_task(struct_elem):
    # Створення структурного елементу "послідовність точок" (розмір: 1x5)
    eroded = binary_erosion(two_task(), structure=struct_elem)
    show_image("Ерозія", eroded)
    return eroded


def four_task(struct_elem):
    # Операція дилатації
    dilated = binary_dilation(three_task(struct_elem), structure=struct_elem)
    show_image("Дилатація", dilated)
    return dilated


def five_task(struct_elem):
    # Повторна ерозія та дилатація для згладжування контурів
    smoothed = binary_erosion(four_task(struct_elem), structure=struct_elem).astype(
        np.uint8
    )
    smoothed = binary_dilation(smoothed, structure=struct_elem)
    show_image("Згладжене зображення", smoothed)
    return smoothed


def six_task(struct_elem):
    # Морфологічне розмикання
    opened = binary_opening(two_task(), structure=struct_elem)
    show_image("Морфологічне розмикання", opened)
    return opened


def seven_task(struct_elem):
    # Морфологічне замикання
    closed = binary_closing(six_task(struct_elem), structure=struct_elem)
    show_image("Морфологічне замикання", closed)


def main():
    struct_elem = np.ones((1, 9), dtype=np.uint8)
    seven_task(struct_elem)
    five_task(struct_elem)


if __name__ == "__main__":
    main()
