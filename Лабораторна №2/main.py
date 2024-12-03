import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, laplace, median_filter, sobel


# Функція для відображення зображення
def show_image(img, title=""):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"images/{title}.jpg")
    plt.close()


# Функція для завантаження зображення та перетворення його в масив numpy
def load_image(image_path):
    img = Image.open(image_path).convert("L")  # Конвертуємо в відтінки сірого
    return np.array(img)


# 1. Перетворення зображення в негатив
def one_task(img):
    return 255 - img


# 2. Логарифмічне перетворення
def two_task(img):
    c = 255 / np.log(1 + np.max(img))  # Нормалізаційний коефіцієнт
    return (c * np.log(1 + img)).astype(np.uint8)


# 3. Степеневе перетворення
def three_task(img, gamma=1.0):
    img_normalized = img / 255.0  # Нормалізація до діапазону [0, 1]
    return (255 * (img_normalized**gamma)).astype(np.uint8)


# 4. Усереднююче згладжування (фільтр середнього)
def four_task(img, size=3):
    return gaussian_filter(img, sigma=size)


# 5. Підвищення різкості за допомогою маски Лапласа
def five_task(img):
    laplacian = laplace(img)  # Обчислення лапласіана
    return np.clip(img - laplacian, 0, 255).astype(np.uint8)


# 6. Медіанна фільтрація
def six_task(img, size=3):
    return median_filter(img, size=size)


# 7. Гаусівське розмиття з різними відхиленнями
def seven_task(img, sigma):
    return gaussian_filter(img, sigma=sigma)


# 8. Нормалізація зображення за допомогою обчислення частки
def eight_task(img, sigma):
    blurred_img = gaussian_filter(img, sigma=sigma)  # Гаусівське розмиття
    return np.clip(img / (blurred_img + 1), 0, 255).astype(np.uint8)


# 9. Пошук контурів за допомогою градієнтів
def nine_task(img):
    grad_x = sobel(img, axis=0)  # Градієнт по осі X
    grad_y = sobel(img, axis=1)  # Градієнт по осі Y
    gradient_magnitude = np.hypot(grad_x, grad_y)  # Обчислення величини градієнта
    return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)


# 10. Нерізке маскування


def ten_task(img, sigma=1, strength=1.5):
    blurred = gaussian_filter(img, sigma=sigma)  # Розмиття
    mask = img - blurred  # Обчислення маски
    return np.clip(img + strength * mask, 0, 255).astype(np.uint8)


def main():
    img = load_image("17.jpg")
    show_image(one_task(img), "Негатив")
    show_image(two_task(img), "Логарифмічне перетворення")
    show_image(three_task(img, gamma=2.0), "Степеневе перетворення (γ=2)")
    show_image(four_task(img), "Усереднююче згладжування")
    show_image(five_task(img), "Підвищення різкості (Лаплас)")
    show_image(six_task(img), "Медіанна фільтрація")

    # Гаусівське розмиття з різними відхиленнями
    for sigma in [2, 5, 10]:
        show_image(seven_task(img, sigma), f"Гаусівське розмиття (σ={sigma})")

    # Нормалізація зображення
    show_image(eight_task(img, sigma=10), "Нормалізація зображення")

    # Пошук контурів
    show_image(nine_task(img), "Контури")

    # Нерізке маскування
    show_image(ten_task(img), "Нерізке маскування")


if __name__ == "__main__":
    main()
