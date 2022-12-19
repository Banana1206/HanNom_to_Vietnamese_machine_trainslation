import json
import string
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

url = 'data/NomNaNMT.csv'

def remove_punctuation_viet(sen):
    """
    :input: sen: str
    :doing:
        1. Xóa dấu câu và số
        2. Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)
    :return:
        Dữ liệu không chứa dấu câu và số
    """
    sen = sen.lower()
    sen = sen.strip()
    sen = re.sub("\[|\]|\.|,|'|:","", sen)
    sen = re.sub("\s+", " ", sen)
    sen = " ".join([s for s in sen.split() if s not in list(string.punctuation)])
    # print(sen)
    return "<sos> " + sen + " <eos>"

def add_space_nom(sen):
    """
    :input: sen: str
    :doing:
        1. Thêm khoảng trắng
        2. Xóa các kí tự ?
        3. Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)
    :return:
        Dữ liệu được ngăn cách bưởi khoảng trắng, bắt đầu bằng <sos> và kết thúc bằng <eos>
    """
    sen = sen.strip()
    sen = ' '.join(sen)
    # sen = re.sub("\[|\]|\.|,|'|:","", sen)
    # sen = re.sub("\s+", " ", sen)
    # sen = " ".join([s for s in sen.split() if s not in list(string.punctuation)])
    return "<sos> " + sen + " <eos>"

class DatasetLoader:
  """
  :input:
        Khởi tạo dữ liệu cho quá trình huấn luyện, bao gồm 2 tập.
        NomNaNMT.csv
            
    :doing:
        1. Khởi tạo liệu
        2. Xóa dấu câu và số
        3. Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)
        4. Xử lý độ dài câu: min_length <= length <= max_length
    :return:
        Dữ liệu sau khi tiền xử lý: list
  """
  def __init__(self, min_length = 4, max_length = 30, url=None):
    #  self.ChuNom = ChuNom
    #  self.TiengViet = TiengViet
     self.min_length = min_length
     self.max_length = max_length
     self.url = url
  
  def load_data(self):
    df = pd.read_csv(url)

    nom_raw = list([nom_raw for nom_raw in df['Nom']])
    viet_raw = list([viet_raw for viet_raw in df['Viet']])

    return nom_raw, viet_raw
  
  def preprocessing_sentence(self, nom_raw, viet_raw):
        """
        :input:
            nom_raw: Ngôn ngữ gốc: chữ nôm
            viet_raw: Ngôn ngữ mục tiêu: chữ tiếng việt
        :doing:
            1. Xử lý độ dài câu: min_length <= length <= max_length
        :return:
        """
        sentences_1 = []
        sentences_2 = []
        for sen_1, sen_2 in zip(nom_raw, viet_raw):
            sen_1 = add_space_nom(sen_1)
            sen_2 = remove_punctuation_viet(sen_2)
            if self.min_length <= len(sen_1.split(" ")) <= self.max_length \
                    and self.min_length <= len(sen_2.split()) <= self.max_length:
              sentences_1.append(sen_1)
              sentences_2.append(sen_2)
            #   print(sentences_1, sentences_2)

        print('Max length is: ',self.max_length)
        return sentences_1, sentences_2

  def build_dataset(self):
    """
        :doing:
            1. Khởi tạo liệu
            2. Xóa dấu câu và số
            3. Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)
            4. Xử lý độ dài câu: min_length <= length <= max_length
        :return:
    """
    nom_raw, viet_raw = self.load_data()
    inp_lang, tar_lang = self.preprocessing_sentence(nom_raw, viet_raw)
    # Build Tokenizer
    tokenize_inp = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenize_tar = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    # Fit text
    tokenize_inp.fit_on_texts(inp_lang)
    tokenize_tar.fit_on_texts(tar_lang)

    # Get tensor
    inp_vector = tokenize_inp.texts_to_sequences(inp_lang)
    tar_vector = tokenize_tar.texts_to_sequences(tar_lang)

    # pad documents to a max length of 20 word
    # Các đầu vào có thể khác nhau nhưng mà input đầu vào cần phải giống nhau => pad_sequences
    max_length = 20
    inp_vector = pad_sequences(inp_vector, maxlen=max_length, padding='post')
    tar_vector = pad_sequences(tar_vector, maxlen=max_length, padding='post')
    return inp_vector, tar_vector ,tokenize_inp, tokenize_tar
  
inp_vector, tar_vector ,tokenize_inp, tokenize_tar = DatasetLoader(url=url).build_dataset()
# print(type(tokenize_inp))

#target_data = tar_vector
#source_data=inp_vector
#target_tokenizer = tokenize_tar
#source_tokenizer = tokenize_inp
