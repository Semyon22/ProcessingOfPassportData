import time
import numpy as np
import cv2
import pytesseract
import imutils
def remove_elements(image):
    """
    Удаляет из изображения объекты небольшого размера

    Параметры
    ---------
    image : numpy.ndarray
        Черно-белое изображение, где фон - черный, а объекты - белые

    Возвращает
    ----------
    numpy.ndarray
        Черно-белое изображение без объектов небольшого размера
    """
    # получаем данные по найденным компонентам
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8)
    # получаем размеры каждой компоненты
    sizes = stats[1:, -1]
    # не рассматриваем фон как компоненту
    nb_components = nb_components - 1
    # определяем минимальный размер компоненты
    min_size = 300
    # формируем итоговое изображение
    image = np.zeros((output.shape))
    # перебираем все компоненты в изображении и сохраняем только тогда, когда она больше min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            image[output == i + 1] = 255
    return image

for i in range(2,32):
    img = cv2.imread(f'passports for reading machine-readable records/{i}.jpg')

    # конвертируем изображение в черно-белый формат и улучшаем качество изображения

    img_gray = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_gray2 = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 25)

    #выделяем области с ФИО
    x, y = img_gray.shape

    x_start=round(x*0.52)
    x_end=round(x*0.71)#горизонталь
    y_start=round(y*0.53)
    y_end = round(y * 0.88)#вертикаль
    img_gray_obr=img_gray[x_start:x_end,y_start:y_end]
    img_gray2_obr=img_gray2[x_start:x_end,y_start:y_end]
    img=img[x_start:x_end,y_start:y_end]
    #тестовые модификаций
    img_gray2_obr = cv2.medianBlur(img_gray2_obr, 3)  # медианная фильтрация
    # img_gray2_obr = cv2.GaussianBlur(img_gray2_obr, (3, 3), 2, 2)
    cv2.imshow(f"img_gray2_obr", img_gray2_obr)
    invert=255-img_gray2_obr
    cv2.imshow(f"invert", invert)
    # определение ядра свертки
    morph_kernel = np.ones((1, 1))
    dilate_img = cv2.dilate(invert, kernel=morph_kernel, iterations=1)
    erode_img = cv2.erode(invert, kernel=morph_kernel, iterations=1)
    cv2.imshow(f"dilate", dilate_img)
    cv2.imshow('erode',erode_img)

    # повышение резкости изображения
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 9, -1],
    #                    [-1, -1, -1]])
    # img_gray2_obr = cv2.filter2D(img_gray2_obr, -1, kernel)
    #вывод областей для отладки
    # cv2.imshow(f"img_gray_obr", img_gray_obr)
    # cv2.imshow(f"img_gray2_obr", img_gray2_obr)
    # cv2.imshow(f"inv", invert1)
    # cv2.imshow(f"img", img)

    #распознование текста в трех вариантах
    config = r'-l rus --oem 1 --psm 6 '
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text_gray = pytesseract.image_to_string(img_gray_obr, lang='ru', config=config)
    print('-------------------------------------------------')
    print('text gray')
    print(text_gray)
    print('-------------------------------------------------')
    print('text gray2')
    text_gray2 = pytesseract.image_to_string(img_gray2_obr, lang='ru', config=config)
    print(text_gray2)
    print('-------------------------------------------------')
    print('text orig')
    text_orig=pytesseract.image_to_string(img, lang='ru', config=config)
    print(text_orig)
    cv2.waitKey()



# y = y * 0.42  # 42
# y = int(round(y))
# x = x * 0.11  # 16
# x = int(round(x))
# img_rot3 = img_rot2[0:x, 0:y]  # чернобелый перевернутый , обрезанный , улучшенный
# img_orig_rot = img_orig_rot[0:x, 0:y]  # оригинал , обрезанный
# img_rot4 = img_rot[0:x, 0:y]  # чернобелый перевернутый обрезанный
#
# cv2.imshow(f"_rot3", img_rot3)
# cv2.imshow(f"1_rot3", img_rot4)

