"""
извлечение текста из паспорта
pip install Pillow
pip install opencv-python
pip install opencv-contrib-python
pip install pytesseract
https://digi.bib.uni-mannheim.de/tesseract/
"""
import re
import time

import cv2
import pytesseract
import imutils
from preprocessing import preprocessing_img
import natasha
from natasha import (
    Segmenter,
    MorphVocab,
    PER,
    NamesExtractor,
    NewsNERTagger,
    NewsEmbedding,
    Doc
)


#todo подумать как можно увеличить качество распознования
#todo написать подпрограмму для распознавания места рождения
#todo попробовать прикрутить готовое решение в случае если своё не отработает
def get_value_series_number(gray,gray1,blur):
    """
    Функция определяющая наиболее вероятный номер и серию паспорта

    :param gray:
    :param gray1:
    :param blur:
    :return:
    """
    result=[0]*10
    #убрать символы пробелов из строк
    gray_nsp,gray1_nsp,blur_nsp=re.sub("[|\n| |]",'',gray),re.sub("[|\n| |]",'',gray1),re.sub("[|\n| |]",'',blur)
    #проверка того что все строки нужной размерности

    if (len(gray_nsp)==10 and len(gray1_nsp)==10 and len(blur_nsp)==10):
        for i in range(0,10):
            if (gray1_nsp[i]==gray_nsp[i]==blur_nsp[i]):
                result[i]=gray1_nsp[i]
            elif(gray1_nsp[i]==gray_nsp[i]):
                result[i]=gray1_nsp[i]
            elif(gray1_nsp[i]==blur_nsp[i]):
                result[i]=gray1_nsp[i]
            elif(gray_nsp[i]==blur_nsp[i]):
                result[i]=gray_nsp[i]
        return "".join(result)
    #находим строки с корректным количеством символов
    elif((len(gray_nsp)==10 and len(gray1_nsp)==10)):
        for i in range(0, 10):
          if (gray1_nsp[i] == gray_nsp[i]):
            result[i] = gray1_nsp[i]
          else: result[i]=gray_nsp[i] #выбираем эту строку поскольку она наиболее достоверна
        return "".join(result)
    elif ((len(gray_nsp) == 10 and len(blur_nsp) == 10)):
        for i in range(0, 10):
          if (gray_nsp[i] == blur_nsp[i]):
            result[i] = gray_nsp[i]
          else: result[i]=blur_nsp[i] #выбираем эту строку поскольку она наиболее достоверна
        return "".join(result)
    elif ((len(gray1_nsp) == 10 and len(blur_nsp) == 10)):
        for i in range(0, 10):
          if (gray1_nsp[i] == blur_nsp[i]):
            result[i] = gray1_nsp[i]
          else: result[i]=blur_nsp[i] #выбираем эту строку поскольку она наиболее достоверна
        return "".join(result)
    #если только одна строка удовлетворяет размерности
    elif(len(gray_nsp)==10):return gray_nsp
    elif(len(gray1_nsp)==10):return gray1_nsp
    elif(len(blur_nsp)==10):return blur_nsp
    else:return "Не удалось обработать серию и номер"
def get_series_number(file_name,debug_marker=0):
    #todo продумать тип возращаемого значения , когда обработка не удалась
    """
    Функция обрабатывающая изображение паспорта , выделяя из него серию и номер
    :param file_name:прописывается путь до файла
    :param debug_marker:ставится 1 если нужен вывод данных на консоль
    :return: строку с серией и номером
    """
    img = cv2.imread(file_name)
    x, y,_ = img.shape

    x_start=int(round(x*0.01))#вертикаль верх
    y_start=int(round(y*0.15))#горизонталь лево
    y = y * 0.6#42 горизонатль право
    y_end = int(round(y))
    x = x * 0.09#16 вертикаль низ
    x_end = int(round(x))
    gray, gray1, blur_gray = preprocessing_img(img, x_start, x_end, y_start, y_end, 270)
    if (debug_marker):
        # распознаем текст с помощью Tesseract
        config = r'-l rus --oem 1 --psm 3 -c tessedit_char_whitelist=0123456789'

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text_gray = pytesseract.image_to_string(gray, lang='ru', config=config)
        text_gray_sp=text_gray[:2] + ' ' + text_gray[2:4] + ' ' + text_gray[4:]
        print(f'passport:{file_name}.jpg gray', text_gray_sp)
        print('----------------------------------------------------------------------------')
        text_gray1 = pytesseract.image_to_string(gray1, lang='ru', config=config)
        text_gray1_sp=text_gray1[:2] + ' ' + text_gray1[2:4] + ' ' + text_gray1[4:]
        print(f'passport:{file_name}.jpg gray1', text_gray1_sp)
        print('----------------------------------------------------------------------------')
        text_blur = pytesseract.image_to_string(blur_gray, lang='ru', config=config)

        text_blur_sp=text_blur[:2]+' '+text_blur[2:4]+' '+text_blur[4:]
        print(f'passport:{file_name}.jpg blur', text_blur_sp)
        print('----------------------------------------------------------------------------')
        print('обработанные значения:')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        result2 = get_value_series_number(text_gray_sp, text_gray1_sp, text_blur_sp)
        result3 = result2[:2] + ' ' + result2[2:4] + ' ' + result2[4:]
        print(result3)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        cv2.imshow(f"{file_name}_gray", gray)
        cv2.imshow(f"{file_name}_gray1", gray1)
        cv2.imshow(f"{file_name}_blur_gray", blur_gray)
        cv2.waitKey(0)
        return result2
    else:
        config = r'-l rus --oem 1 --psm 3 -c tessedit_char_whitelist=0123456789'
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text_gray = pytesseract.image_to_string(gray, lang='ru', config=config)
        text_gray_sp = text_gray[:2] + ' ' + text_gray[2:4] + ' ' + text_gray[4:]
        text_gray1 = pytesseract.image_to_string(gray1, lang='ru', config=config)
        text_gray1_sp = text_gray1[:2] + ' ' + text_gray1[2:4] + ' ' + text_gray1[4:]
        text_blur = pytesseract.image_to_string(blur_gray, lang='ru', config=config)
        text_blur_sp = text_blur[:2] + ' ' + text_blur[2:4] + ' ' + text_blur[4:]
        result = get_value_series_number(text_gray_sp, text_gray1_sp, text_blur_sp)
        return result
def string_processing(str):
    """
        Эта функция обрабатывает строку и выделяет из неё результирующую строку с предполагаемым фио
    """
    str1 = str.lower()
    text = re.sub("[1|2|3|4|5|6|7|8|9|0|.|,|/|:|<|>|e|?|©|=|&|_|@|\|“|+|-|„|^|]", '', str1)
    list = text.split(' ')
    result = ''
    for i in range(0, len(list)):
        new_list = list[i].split('\n')
        list[i] = new_list
    for i in range(0, len(list)):
        #исключение из множества слова с двумя буква
        for j in range(0, len(list[i])):
            #Делаем первую букву заглавной
            list[i][j] = list[i][j].capitalize()
            if len(list[i][j]) < 3:
                list[i][j] = ''
    for item in list:
        # исключение из результирующей строки слов с тремя одинаковыми буквами идущими подряд
        for j in range(0, len(item)):
            counter = 1
            for k in range(0, len(item[j]) - 1):
                if item[j][k] == item[j][k + 1]:
                    counter += 1
                else:
                    counter = 1
                if counter == 3:
                    item[j] = ''
        #объеденение элементов листа в одну строку
        result = result + ' ' + ' '.join(item)
    # обрезание пробелов с конца и начала
    result1 = result.lstrip().rstrip()
    return result1
def natasha_help(str):
    """
    распознование фио при помощи библиотеки наташа
    :param str:
    :return:
    """
    emb = NewsEmbedding()
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    ner_tagger = NewsNERTagger(emb)
    names_extractor = NamesExtractor(morph_vocab)
    text = str  # текст добавляем сюда
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    for span in doc.spans:
        span.normalize(morph_vocab)
    {_.text: _.normal for _ in doc.spans}
    for span in doc.spans:
        if span.type == PER:
            span.extract_fact(names_extractor)
    dict = {'None': _.fact.as_dict for _ in doc.spans if _.fact}
    if 'None' in dict.keys():
        if 'middle' not in dict['None'].keys(): dict['None']['middle'] = None
        if 'last' not in dict['None'].keys(): dict['None']['last'] = None
        if 'first' not in dict['None'].keys(): dict['None']['first'] = None
        return dict
    else:return {'None':{'first':None , 'middle':None , 'last' : None }}
def sampl_fio(gray_dict, gray1_dict, blur_dict):
    """
    Извлекает из трёх словарей корректное фио
    :param gray_dict:
    :param gray1_dict:
    :param blur_dict:
    :return:
    """
    result = {'first': '', 'last': '', 'middle': ''}
    gray_dict = gray_dict['None']
    gray1_dict = gray1_dict['None']
    blur_dict = blur_dict['None']
    for key in result:
        if gray_dict[key] == gray1_dict[key] == blur_dict[key]:
            result[key] = gray1_dict[key]
        elif (gray_dict[key] == gray1_dict[key]) and (gray_dict[key]!=None):
            result[key] = gray_dict[key]
        elif gray_dict[key] == blur_dict[key] and (gray_dict[key]!=None):
            result[key] = blur_dict[key]
        elif gray1_dict[key] == blur_dict[key] and (gray1_dict[key]!=None):
            result[key] = gray1_dict[key]
        else:
            if gray_dict[key]!=None:
                result[key] = gray_dict[key]
            elif gray1_dict[key]!=None:
                result[key]=gray1_dict[key]
            elif blur_dict[key]!=None:
                result[key]=blur_dict[key]
    return result
def get_fio(img, debug_marker=0):
    """

    :param img:Путь до файла
    :param debug_marker: Если нужен вывод данных на консоль
    :return: словарь с фио
    """
    # выделяем области с ФИО
    img = cv2.imread(img)
    x, y,_ = img.shape
    x_start = round(x * 0.52)
    x_end = round(x * 0.71)  # горизонталь
    y_start = round(y * 0.50)
    y_end = round(y * 0.88)  # вертикаль
    gray, gray1, blur_gray = preprocessing_img(img,x_start,x_end,y_start,y_end,0)
    config = r'-l rus --oem 1 --psm 6 '
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text_gray = pytesseract.image_to_string(gray, lang='ru', config=config)
    if (debug_marker):
        print('-------------------------------------------------')
        print('gray')
        # print(text_gray)
        print(string_processing(text_gray))
        k = string_processing(text_gray)
        print(natasha_help(k))
        gray_dict = natasha_help(k)

        print('-------------------------------------------------')
        print('gray1')
        text_gray1 = pytesseract.image_to_string(gray1, lang='ru', config=config)
        print(text_gray1)
        print(string_processing(text_gray1))
        k = string_processing(text_gray1)
        print(natasha_help(k))
        gray1_dict = natasha_help(k)
        print('-------------------------------------------------')
        print('blured')
        blured_text = pytesseract.image_to_string(blur_gray, lang='ru', config=config)
        print(blured_text)
        print(string_processing(blured_text))
        k = string_processing(blured_text)
        print(natasha_help(k))
        blur_dict = natasha_help(k)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(sampl_fio(gray_dict, gray1_dict, blur_dict))
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        cv2.imshow('gray', gray)
        cv2.imshow('gray1', gray1)
        cv2.imshow('blur_gray', blur_gray)
        cv2.waitKey(0)
    else:
        k = string_processing(text_gray)
        gray_dict = natasha_help(k)
        text_gray1 = pytesseract.image_to_string(gray1, lang='ru', config=config)
        k = string_processing(text_gray1)
        gray1_dict = natasha_help(k)
        blured_text = pytesseract.image_to_string(blur_gray, lang='ru', config=config)
        k = string_processing(blured_text)
        blur_dict = natasha_help(k)
        return  sampl_fio(gray_dict, gray1_dict, blur_dict)
def final_processing_birthdate(gray,gray1,blur):
    """
    Функция находит наиболее вероятную дату рождения
    путём обработки трёх строк и проверки на корректность их содержимого
    :param gray:str
    :param gray1:str
    :param blur:str
    :return: str result
    """
    def delete_dots(str):
        """
        Функция удаляет из строки лишние точки в конце и в начале и
        возвращает список созданный из строки путем разбиения по точкам
        :param str:
        :return:
        """
        #если прилетела пустая строка
        if (len(str)!=0):
            # если считались лишние точки
            if str[0] == '.' or str[len(str)-1] == '.':
                # если лишние точки в конце и в начале
                if str[0] == '.' and str[len(str)-1] == '.':
                    str_buf = str[1:len(str)-1]
                    arr = str_buf.split('.')
                # если лишняя точка в начале
                elif str[0] == '.':
                    str_buf = str[1:]
                    arr = str_buf.split('.')
                # если лишняя точка в конце
                elif str[len(str)-1] == '.':
                    str_buf = str[:len(gray_nsp)-1]
                    arr = str_buf.split('.')
            else:
                arr = str.split('.')
            return arr
        else: return ['0','0','0']
    try:
        #обрезка лишних пробелов
        gray_nsp,gray1_nsp,blur_nsp=re.sub("[|\n| |]", '', gray),re.sub("[|\n| |]", '', gray1),re.sub("[|\n| |]", '', blur)
        if gray1_nsp==gray_nsp==blur_nsp:return gray1_nsp # случай совпадения всех строк
        arr=[['0','0','0']]*3
        arr[0]=delete_dots(gray_nsp)
        arr[1]=delete_dots(gray1_nsp)
        arr[2]=delete_dots(blur_nsp)
        #цикл проверки корректности входных данных
        for i in range(0,len(arr)):
            #цикл в котором строка наращивается до фиксированного размера
            while len(arr[i])<3:
                arr[i].append('0')
            if int(arr[i][0])>31 or int(arr[i][1])>12 or int(arr[i][2])<1956 or 2010<int(arr[i][2]):
                arr[i].append('uncorrectly')
            else:
                arr[i].append('correctly')
        # print(arr) в случае отладки расскоментировать
        if (arr[0][3]==arr[1][3]=='correctly') and arr[0][0]==arr[1][0] and arr[0][1]==arr[1][1] and arr[0][2]==arr[1][2]:
                return '.'.join(arr[0][0:3])
        elif (arr[0][3]==arr[2][3]=='correctly') and arr[0][0]==arr[2][0] and arr[0][1]==arr[2][1] and arr[0][2]==arr[2][2]:
            return '.'.join(arr[0][0:3])
        elif (arr[1][3]==arr[2][3]=='correctly') and arr[1][0]==arr[2][0] and arr[1][1]==arr[2][1] and arr[1][2]==arr[2][2]:
            return '.'.join(arr[1][0:3])
        #если только один корректный результат
        for i in range(0,len(arr)):
            if arr[i][3]=='correctly':return '.'.join(arr[i][0:3])
    except Exception as Ex:
        print(Ex)
def get_birthdate(file_name,debug_marker=0):
    """
    В данной функций извелкается дата рождения
    :param file_name:
    :return:
    """
    #todo рассмотреть случай когда нейронка считывает сначала мусор , а через пробел нужную инфу
    img = cv2.imread(file_name)
    #получение размера изображения
    x, y, _ = img.shape
    #выделение области с датой рождения
    x_start = int(round(x * 0.68))  # вертикаль верх
    x_end = int(round(x * 0.75))  # вертикаль низ
    y_start = int(round(y * 0.59))  # горизонталь лево
    y_end = int(round(y * 0.89))# горизонталь право


    #предобработка изображений
    gray,gray1,blur_gray=preprocessing_img(img,x_start,x_end,y_start,y_end,0)
    #Подключение OCR

    config = r'-l rus --oem 1 --psm 3 -c tessedit_char_whitelist=0123456789.'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #Получение текста с фото
    text_gray = pytesseract.image_to_string(gray, lang='ru', config=config)
    text_gray1 = pytesseract.image_to_string(gray1, lang='ru', config=config)
    text_blur = pytesseract.image_to_string(blur_gray, lang='ru', config=config)
    #Финальная обработка текста
    if (debug_marker):
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(final_processing_birthdate(text_gray,text_gray1,text_blur))
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        #вывод полученных изображений
        print(f"{file_name}_gray" ,":", text_gray)
        cv2.imshow(f"{file_name}_gray", gray)
        print(f"{file_name}_gray1", ":", text_gray1)
        cv2.imshow(f"{file_name}_gray1", gray1)
        print(f"{file_name}_blur_gray", ":", text_blur)
        cv2.imshow(f"{file_name}_blur_gray", blur_gray)
        cv2.waitKey(0)
    else: return final_processing_birthdate(text_gray,text_gray1,text_blur)
def get_birthplace(file_name):
    img = cv2.imread(file_name)
    # получение размера изображения
    x, y, _ = img.shape
    # выделение области с датой рождения
    x_start = int(round(x * 0.737))  # вертикаль верх
    x_end = int(round(x * 0.870291))  # вертикаль низ
    y_start = int(round(y * 0.339398))  # горизонталь лево
    y_end = int(round(y * 0.885786))  # горизонталь право

    # предобработка изображений
    gray, gray1, blur_gray = preprocessing_img(img, x_start, x_end, y_start, y_end, 0)
    # Подключение OCR

    config = r'-l rus --oem 1 --psm 3 -c tessedit_char_blacklist=,?!:;-()<>/|[]{}_`'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Получение текста с фото
    text_gray = pytesseract.image_to_string(gray, lang='ru', config=config)
    text_gray1 = pytesseract.image_to_string(gray1, lang='ru', config=config)
    text_blur = pytesseract.image_to_string(blur_gray, lang='ru', config=config)
    list1 = text_gray.replace('\n','').split(' ')
    print(f"{file_name}_gray", ":", text_gray)

    cv2.imshow(f"{file_name}_gray", gray)
    list2=text_gray1.replace('\n','').split(' ')
    print(f"{file_name}_gray1", ":", text_gray1)

    cv2.imshow(f"{file_name}_gray1", gray1)
    print(f"{file_name}_blur_gray", ":", text_blur)
    list3=text_blur.replace('\n','').split(' ')
    print(f"{file_name}_gray", ":", list1)
    print(f"{file_name}_gray1", ":", list2)
    print(f"{file_name}_blur_gray", ":", list3)
    cv2.imshow(f"{file_name}_blur_gray", blur_gray)
    cv2.waitKey(0)
def get_full_pass_data(file_name):
    """
    Функция для получения полной информаций по фото паспорта
    :param file_name: str
    :return: dict
    """
    result={"series":'','number':'','first_name':'','last_name':'','patronymic':'','birthdate':''}
    series_number=get_series_number(file_name)
    result['series']=series_number[0:4]
    result['number']=series_number[4:]
    fio=get_fio(file_name)
    result['first_name']=fio['first']
    result['last_name']=fio['last']
    result['patronymic']=fio['middle']
    birthdate=get_birthdate(file_name)
    result['birthdate']=birthdate
    return result
for i in range(1,13):
    img = cv2.imread(f'pasports_fio/{i}.jpg')
    cv2.imshow('img', img)

    print(get_full_pass_data(f'pasports_fio/{i}.jpg'))

    # print(get_birthplace(f'pasports_fio/{i}.jpg'))
    cv2.waitKey(0)
exit()
