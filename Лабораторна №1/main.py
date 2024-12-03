from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np


# Вірізаемо зображення 18*18
def one_task(img: Image):
    change_img = img.crop((0, 0, 18, 18))
    change_img.save("one_task.jpg")


# Змінюемо розмір та повертаем на 180
def two_task(img: Image):
    change_img = img.rotate(180)
    change_img = change_img.resize((1920, 1080))
    change_img.save("two_task.jpg")


# Робимо чорно-біле зображення
def three_task(img: Image):
    change_img = img.convert("L")
    change_img.save("three_task.jpg")


# Малюемо точки та лыныъ на зображенні
def four_task(img):
    im = pylab.array(img)
    pylab.imshow(im)
    x = [0, 50, 100, 150, 200]
    y = [0, 50, 100, 150, 175]
    plt.plot(x, y[::-1], linestyle="--")
    plt.scatter(x, y[::-1], color="red", label="Точки")

    plt.title("Графік із точками та лініями")
    plt.legend()
    plt.grid(True)
    plt.savefig("four_task.jpg")


# Робимо гістрограми
def five_task(img):
    im = pylab.array(img.convert("L"))
    plt.figure()
    plt.gray()
    plt.contour(im, origin="image")
    plt.axis("equal")
    plt.axis("off")
    plt.figure()
    plt.hist(im.flatten(), 128)
    plt.savefig("five_task.jpg")


def six_task(img):
    matrix = np.array(img)
    matrix[25] = matrix[50]
    Image.fromarray(matrix).save("six_task.jpg")


def seven_task(img):
    matrix = np.array(img)
    matrix[:, 75] = 200
    Image.fromarray(matrix).save("seven_task.jpg")


def eight_task(img):
    matrix = np.array(img)
    sum_row = np.sum(matrix[:50, :])
    sum_col = np.sum(matrix[:, :50])
    print(f"Сума стовпців:{sum_col}")
    print(f"Сума рядків:{sum_row}")


def nine_task(img):
    matrix = np.array(img)
    average = np.average(matrix[50])
    print(f'average:"{average}')


def ten_task(img):
    matrix = np.array(img)
    print(matrix[:-3])


#  еквілізація гістрограми
def eleven_task(img):
    im = np.array(img.convert("L"))
    nbr_bins = 256
    imhist, bins = np.histogram(im.flatten(), nbr_bins)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    im3 = im2.reshape(im.shape)
    plt.imshow(im3)
    plt.axis("off")
    plt.savefig("eleven_task.jpg")


def main():
    img = Image.open("17.jpg")
    one_task(img), two_task(img), three_task(img)
    four_task(img)
    five_task(img)
    six_task(img)
    seven_task(img)
    eight_task(img)
    nine_task(img)
    ten_task(img)
    eleven_task(img)


if __name__ == "__main__":
    main()
