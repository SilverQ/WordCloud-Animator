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
font_path = 'NanumGothic.ttf'
train = pd.read_csv(input_file, delimiter='\t')
print(f'total data length: {len(train)}')
# print(train.head())

sentences = list(train['발명의명칭'])
# print(sentences[1])

# okt = Okt()
# tokenized_by_space = [s.split() for s in sentences]
# tokenized_by_morph = [okt.morphs(s) for s in sentences]
# tokenized_by_char = [list(s) for s in sentences]

stopwords = set(STOPWORDS)
stopwords.add('방법')
stopwords.add('위한')
stopwords.add('제조')
stopwords.add('제조방법')
stopwords.add('장치')
# stopwords.add('방법')
# stopwords.add('방법')
# stopwords.add('방법')
# stopwords.add('방법')

img_path = 'img/'
img_list = []
img_fn_list = []

for i in range(10):
    input = ' '.join(sentences[1+i:30+i])
    wc = WordCloud(font_path=font_path, stopwords=stopwords,
                   max_words=100, random_state=42).generate(input)
    # wc.generate(tokenized_by_space)
    f = plt.figure(figsize=(20, 20))
    img_list.append(f)
    plt.imshow(wc)
    plt.axis("off")
    img_fn_list.append(img_path+'wordcloud_'+str(i)+'.png')
    plt.savefig(img_path+'wordcloud_'+str(i)+'.png')


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

images = [np.array(imageio.imread(img)) for img in img_fn_list]
imageio.mimsave('gif/animation3.gif', images, fps=3)
