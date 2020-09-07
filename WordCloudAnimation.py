import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import imageio
# from PIL import Image
# import csv
# import pickle
# from nltk.tokenize import word_tokenize
# import os
# import csv
# import time
# import spacy
# import psycopg2 as pg2
# !pip install konlpy
# !pip install -q wordcloud
# from konlpy.tag import Okt
# !pip install python-levenshtein
# !pip install -q wordcloud
# import nltk
# import wordcloud
# import Levenshtein
# nltk.download('punkt')
# %matplotlib inline
# !pip install imageio
# import glob

# base_path
# input_file = base_path + 'test_data.txt'
input_file = 'test_data.txt'

# 전체 컬럼 : 특허번호	출원번호	출원접수국가	출원일	국제특허분류(IPC)	발명의명칭	출원인/특허권자	요약	대표청구항	과제	해결방안
# 정렬 컬럼 : 출원일
# 필터 컬럼 :
# 입력 문자열 :  발명의명칭, 출원인/특허권자, 요약, 대표청구항, 과제, 해결방안

font_path = 'NanumGothic.ttf'
train = pd.read_csv(input_file, delimiter='\t')
train.sort_values(['출원일'], ascending=[True])
print(f'total data length: {len(train)}')
print(train.head())

# sentences = list(train['발명의명칭'])
sentences = list(train['요약'])
print(sentences[1])

# okt = Okt()
# tokenized_by_space = [s.split() for s in sentences]
# tokenized_by_morph = [okt.morphs(s) for s in sentences]
# tokenized_by_char = [list(s) for s in sentences]

# print(STOPWORDS)

img_path = 'img/'
stopwords = set(STOPWORDS)
stopwords.add('방법')
stopwords.add('위한')
stopwords.add('제조')
stopwords.add('제조방법')
stopwords.add('장치')
stopwords.add('상기')
stopwords.add('발명은')
stopwords.add('또는')
# stopwords.add('방법')
img_list = []
img_fn_list = []

# for i in range(50):
#     input = ' '.join(sentences[1 + i:30 + i])
#     wc = WordCloud(font_path=font_path, stopwords=stopwords,
#                    max_words=150, random_state=42,
#                    width=640, height=480).generate(input)
#     # wc.generate(tokenized_by_space)
#     f = plt.figure(figsize=(20, 10))
#     img_list.append(f)
#     plt.imshow(wc)
#     plt.axis("off")
#     img_fn_list.append(img_path + 'wordcloud_' + str(i) + '.png')
#     plt.savefig(img_path + 'wordcloud_' + str(i) + '.png')


def create_wordcloud(text, filter=3, stride=1, padding=0):
    # len_img = len(sentences)-1
    for i in range(0, len(text)-filter-1, stride):
        input = ' '.join(text[1+i:filter+i])
        wc = WordCloud(font_path=font_path, stopwords=stopwords,
                       max_words=150, random_state=42,
                       width=640, height=480).generate(input)
        # wc.generate(tokenized_by_space)
        f = plt.figure(figsize=(20, 10))
        img_list.append(f)
        plt.imshow(wc)
        plt.axis("off")
        img_fn_list.append(img_path+'wordcloud_'+str(i)+'.png')
        plt.savefig(img_path+'wordcloud_'+str(i)+'.png')
    return img_fn_list, img_list


# def create_animation(images, fpath):
#     images = [np.array(img) for img in images]
#     imageio.mimsave(fpath, images, fps=3)
#     # with open(fpath,'rb') as f:
#     #     display.display(display.Image(data=f.read(), height=512))
#
#
# images = []
# for img in img_fn_list:
#     images.append(imageio.imread(img))
# images = [imageio.imread(img) for img in img_fn_list]
# create_animation(images, 'gif/animation3.gif')

# images = create_wordcloud(sentences[:10], filter=3, stride=1)[1]
img_fn_list = create_wordcloud(sentences[:20], filter=5, stride=5)[0]
images = [np.array(imageio.imread(img)) for img in img_fn_list]
imageio.mimsave('gif/animation_abst.gif', images, fps=2)
#
# images = [np.array(img) for img in img_list]
# imageio.mimsave('gif/animation_abst.gif', images, fps=2)
