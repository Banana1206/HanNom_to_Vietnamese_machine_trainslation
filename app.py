from flask import Flask, redirect, url_for, render_template, request
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

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    return render_template('base.html')
@app.route("/translator",  methods=['GET', 'POST'])
def translator():
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model=Transformer()
    print(latest)
    model.load_weights(latest).expect_partial()
    if request.method == "POST":
        # if request.form['routing'] == "Come back":
        #     return redirect('/')
        nom = request.form['nom']
        print(' '.join(translate(model,' '.join(list(nom)).split(' '))))
        dich = ' '.join(translate(model,' '.join(list(nom)).split(' ')))
    return render_template('translate.html', dich=dich, nom=nom)

if __name__ == "__main__":
    app.run(debug=True)
