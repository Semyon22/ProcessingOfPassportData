import rupasportread as pr
import pytesseract
# pr.download('C:\\Users\\User\\PycharmProjects\\ProcessingOfPassportData\\new pasports', '1.jpg')
for i in range(1,10):
    try:
        pr.catching(f'pasports_fio/{i}.jpg')
    except:
        print("ex",f'{i}')
import natasha
# from natasha import (
#     Segmenter,
#     MorphVocab,
#     PER,
#     NamesExtractor,
#     NewsNERTagger,
#     NewsEmbedding,
#     Doc
# )
# emb = NewsEmbedding()
# segmenter = Segmenter()
# morph_vocab = MorphVocab()
# ner_tagger = NewsNERTagger(emb)
# names_extractor = NamesExtractor(morph_vocab)
#
# text = 'Абдухалимова Елена Григорьевна'  # текст добавляем сюда
#
# doc = Doc(text)
# doc.segment(segmenter)
# doc.tag_ner(ner_tagger)
# for span in doc.spans:
#     span.normalize(morph_vocab)
#     {_.text: _.normal for _ in doc.spans}
# for span in doc.spans:
#     if span.type == PER:
#             span.extract_fact(names_extractor)
#
# print({_.normal: _.fact.as_dict for _ in doc.spans if _.fact})
# blur_dict={'Милонов   Сергей Петрович  Гуа': {'last': 'Милонов'}}
# gray1_dict={'Рне   Нее   Милонов   Сергей Петрович    Пло Лоаа': {'last': 'Милонов'}}
# gray_dict={'Милонов   Сергей Петрович      Лосс': {'first': 'Сергей', 'last': 'Милонов', 'middle': 'Петрович'}}
# print(blur_dict.values())