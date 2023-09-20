import cv2
import pytesseract
import imutils
for i in range(28,29):
    img = cv2.imread(f'new pasports/{i}.jpg')
    # img_orig_rot = imutils.rotate_bound(img, angle=270)
    # img_gray2 = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                   cv2.THRESH_BINARY, 199, 25)
    # img_rot2 = imutils.rotate_bound(img_gray2, angle=270)
    # img_rot3 = img_rot2[0:300 , 0:750]
    print(img.shape)
    gorizontal=img.shape[0]
    vertical=img.shape[1]
    gorizontal_coord_start=int(round(gorizontal*0.53))
    gorizontal_coord_end=int(round(gorizontal*0.9))
    vertical_coord_star=int(round(vertical*0.33))
    vertical_coord_end=int(round(vertical*0.85))
    print(gorizontal,vertical)
    img1_obr=img[gorizontal_coord_start:gorizontal_coord_end,vertical_coord_star:vertical_coord_end]
    img_bw = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 25)
    img_bw2 = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_bw=img_bw[gorizontal_coord_start:gorizontal_coord_end,vertical_coord_star:vertical_coord_end]
    img_bw2=img_bw2[gorizontal_coord_start:gorizontal_coord_end,vertical_coord_star:vertical_coord_end]
    cv2.imshow("orig", img)
    cv2.imshow("obr_orig", img1_obr)
    cv2.imshow("obr_orig1", img_bw)
    cv2.imshow("obr_orig2", img_bw2)
    # cv2.imshow("orig_rot", img_orig_rot)
    # cv2.imshow("Gray rot", img_rot3)
    # cv2.imshow("Gray obr", img_rot2)
    config = r'-l rus --oem 1 --psm 3 tessedit_char_whitelist=0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text_img1_obr = pytesseract.image_to_string(img1_obr, lang='ru', config=config)
    text_img_bw = pytesseract.image_to_string( img_bw, lang='ru', config=config)
    text_img_bw2 = pytesseract.image_to_string(img_bw2, lang='ru', config=config)
    # print('img1_obr:', text_img1_obr)
    # print('-----------------------------------------------')
    # print('img_bw:', text_img_bw)
    # print('-----------------------------------------------')
    # print('text_img_bw2:', text_img_bw2)
    # print('-----------------------------------------------')
    print('----------------------------------------')
    list1 = text_img1_obr.replace('\n','').split(' ')
    print(list1)
    print('----------------------------------------')
    list2 = text_img_bw.replace('\n','').split(' ')
    print(list2)
    print('----------------------------------------')
    list3 = text_img_bw2.replace('\n','').split(' ')
    print(list3)

    cv2.waitKey(0)
