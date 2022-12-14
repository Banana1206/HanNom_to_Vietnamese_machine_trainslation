import numpy as np
import unicodedata, re
import time, os
from preprocessing import DatasetLoader
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization
from Transformer import *
url = 'data/NomNaNMT.csv'
inp_vector, tar_vector ,tokenize_inp, tokenize_tar = DatasetLoader(url=url).build_dataset()
# source_sentences,target_sentences = DatasetLoader(url=url).preprocessing_sentence()
target_labels = np.zeros(tar_vector.shape)
target_labels[:,0:tar_vector.shape[1] -1] = tar_vector[:,1:]
source_vocab_len = len(tokenize_inp.word_index) + 1
target_vocab_len = len(tokenize_tar.word_index) + 1
print("Size of source vocabulary: ", source_vocab_len)
print("Size of target vocabulary: ", target_vocab_len)

dataset = tf.data.Dataset.from_tensor_slices((inp_vector, tar_vector, target_labels)).batch(5)
# For Keras model.fit()
dataset_2 = tf.data.Dataset.from_tensor_slices((inp_vector, tar_vector, target_labels))

d_model = 512 # 512 in the original paper
d_k = 64 # 64 in the original paper
d_v = 64 # 64 in the original paper
n_heads = 8 # 8 in the original paper
n_encoder_layers = 6 # 6 in the original paper
n_decoder_layers = 6 # 6 in the original paper
max_token_length = 20 # 512 in the original paper

# Testing if the dimension matches!
# x = tf.ones((3, 26, d_model))
# x1 = tf.ones((3, 18, d_model))
# single_att = SingleHeadAttention(masked=None)
# multi_att = MultiHeadAttention()
# encoder = TransformerEncoder()
# decoder = TransformerDecoder()
# y = single_att((x, x, x)) # Self attention
# y1 = multi_att((x1, x, x)) # Encoder-decoder attention
# print(tf.shape(y))
# print(tf.shape(y1))
# y2 = encoder(x)
# y3 = decoder(x, y2)

# print(tf.shape(y2))
# print(tf.shape(y3))
#print(layer.trainable_weights)


# Demonstration on calling transformer model
transformer = Transformer(dropout=.1)
print(tf.shape(transformer([np.ones((5, 15)), np.ones((5, 12))], training=False)))


# Specify loss, optimizer and training function
checkpoint_path = "saved_models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

transformer_2 = Transformer() # Instantiating a new transformer model
src_seqs, tgt_seqs, tgt_labels = zip(*dataset_2)
print(tgt_seqs)
train = [tf.cast(src_seqs, dtype=tf.float32), tf.cast(tgt_seqs, dtype=tf.float32)] # Cast the tuples to tensors
print('TRAIN \n',train)
transformer_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
transformer_2.fit(train, tf.cast(tgt_labels, dtype=tf.float32), verbose=1, batch_size=5, epochs=200, callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'), 
                                                                                                              tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=75*batch_size)])

# print("Source sentence: ", source_sentences[10])
# print("Target sentence: ", target_sentences[10])
# print("Predicted sentence: ", ' '.join(translate(transformer, source_sentences[10].split(' '))))

# class ExportTranslator(tf.Module):
#   def __init__(self, translator):
#     self.translator = translator

#   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
#   def __call__(self, sentence):
#     (result,
#      tokens,
#      attention_weights) = self.translator(sentence, max_length=30)

    # return result

# translator = ExportTranslator(transformer_2)
# tf.saved_model.save(translator, export_dir='translator')

# Include the epoch in the file name (uses `str.format`)


# Create a callback that saves the model's weights every 5 epochs
# Create a new model instance
# model = transformer_2

# Save the weights using the `checkpoint_path` format
# transformer_2.save_weights(checkpoint_path.format(epoch=0))
# latest = tf.train.latest_checkpoint(checkpoint_dir)

# model=Transformer()
# model.load_weights(latest)
# print("Source sentence: ", '壬戌元年')
# print("Target sentence: ", 'Nhâm Tuất nguyên niên')
# print("Predicted sentence: ", ' '.join(translate(model, '壬戌元年'.split(' '))))