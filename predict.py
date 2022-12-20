import numpy as np
import unicodedata, re
import time, os
from preprocessing import DatasetLoader
import tensorflow as tf
import pandas as pd
from Transformer import *
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
checkpoint_path = "saved_models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def translate(model, source_sentence, target_sentence_start=[['<sos>']]):
    if np.ndim(source_sentence) == 1: # Create a batch of 1 the input is a sentence
        source_sentence = [source_sentence]
    if np.ndim(target_sentence_start) == 1:
        target_sentence_start = [target_sentence_start]
    # Tokenizing and padding
    source_seq = tokenize_inp.texts_to_sequences(source_sentence)
    source_seq = tf.keras.preprocessing.sequence.pad_sequences(source_seq, padding='post', maxlen=30)
    predict_seq = tokenize_tar.texts_to_sequences(target_sentence_start)
    
    predict_sentence = list(target_sentence_start[0]) # Deep copy here to prevent updates on target_sentence_start
    while predict_sentence[-1] != '<eos>' and len(predict_seq) < max_token_length:
        predict_output = model([np.array(source_seq), np.array(predict_seq)], training=None)
        predict_label = tf.argmax(predict_output, axis=-1) # Pick the label with highest softmax score
        predict_seq = tf.concat([predict_seq, predict_label], axis=-1) # Updating the prediction sequence
        predict_sentence.append(tokenize_tar.index_word[predict_label[0][0].numpy()])
    return predict_sentence[1:-1]
latest = tf.train.latest_checkpoint(checkpoint_dir)
model=Transformer()
print(latest)
model.load_weights(latest).expect_partial()
# source = '大 越 史 記 外 紀 全 書 卷 之'
data = pd.read_csv('data/NomNaNMT.csv')
nom_data = data['Nom'].tolist()
nom_data = nom_data[-500:]
viet_data = data['Viet'].tolist()
viet_data = viet_data[-500:]
reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print('score: ', score)
pred_l, viet_l = [], []

for nom,viet in zip(nom_data, viet_data):
    viet = re.sub("\[|\]|\.|,|'|:","", viet)
    viet = re.sub("\s+", " ", viet)
    # print("Source sentence: ", nom)
    # print("Target sentence: ", viet)
    # print("Predicted sentence: ", ' '.join(translate(model,' '.join(list(nom)).split(' '))))
    pred = ' '.join(translate(model,' '.join(list(nom)).split(' ')))
    pred_sc =pred.split(' ')
    pred_l.append(pred_sc)
    viet_sc = viet.split(' ')
    viet_t = [viet.split(' ')]
    viet_l.append(viet_sc)
    # print('BLEU score: ', sentence_bleu(references=viet_t, hypothesis=pred_sc))

print('Corpus BLEU score: ', corpus_bleu(viet_l, pred_l))

