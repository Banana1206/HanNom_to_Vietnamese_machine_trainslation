import json
import os
import tensorflow as tf
from argparse import ArgumentParser
from preprocessing import remove_punctuation_viet
from metric import CustomSchedule
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Encoder_Attention_decoder import Encode, Attention, Decoder
from preprocessing import DatasetLoader, add_space_nom, remove_punctuation_viet
import pandas as pd

class PredictionSentence_HanNom_to_Viet:
    def __init__(self,
                 embedding_size=64,
                 hidden_units=64,
                 max_sentence=30,
                 learning_rate=0.002):

        home = os.getcwd()
        self.max_sentence = max_sentence
        self.url = 'data/train_dev.csv'

        self.inp_tensor, self.tar_tensor, self.inp_builder, self.tar_builder = DatasetLoader(url=self.url).build_dataset()
        self.values = list(self.tar_builder.index_word.values())
        self.keys = list(self.tar_builder.index_word.keys())

        # Initialize Seq2Seq model
        input_vocab_size = len(self.inp_builder.word_index) + 1
        target_vocab_size = len(self.tar_builder.word_index) + 1

        # input_vocab_size = 3874
        # target_vocab_size = 3874

        # Initialize encoder
        self.encoder = Encode(input_vocab_size,
                              embedding_size,
                              hidden_units)

        # Initialize decoder
        self.decoder = Decoder(target_vocab_size,
                                  embedding_size,
                                  hidden_units)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # Initialize translation
        self.path_save = home + "/saved_models_HanToviet2"
        self.checkpoint_prefix = os.path.join(self.path_save, "Seq2Seq_hanmon_to_VN")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save)).expect_partial()

    def __preprocess_input_text__(self, text):
        inp_lang = add_space_nom(text)
        vector = [[self.inp_builder.word_index[w] for w in inp_lang.split() if w in list(self.inp_builder.word_index.keys())]]
        sentence = pad_sequences(vector,
                                 maxlen=self.max_sentence,
                                 padding="post",
                                 truncating="post")
        # print(sentence)
        return sentence, vector

    def translate_enroll(self, input_text, label):
        vector, sentence = self.__preprocess_input_text__(input_text)
        # Encoder
        encode_outs, last_state = self.encoder(vector)
        # Process decoder input
        input_decode = tf.constant([self.tar_builder.word_index['<sos>']])
        pred_sentence = ""

        # pre_list = []
        for _ in range(self.max_sentence):
            output, last_state = self.decoder(input_decode, encode_outs, last_state)
            pred_id = tf.argmax(output, axis=1).numpy()
            # print('output: ',output)
            # print('pred_id: ',pred_id)
            # print('pred_id type : ',type(pred_id))
            input_decode = pred_id
            word = self.values[pred_id[0]]
            # pre_list.append(pred_id)
            # print(' word : ', word)
            if word not in ["<sos>", "<eos>"]:
                pred_sentence += " " + word
            if word in ["<eos>"]:  
                break
        print("-----------------------------------------------------------------")
        print("Input     : ", input_text)
        print("Label     : ", label)
        print("Translate :", pred_sentence)
        # print('Prelist   :', pre_list)
        print("-----------------------------------------------------------------")
        
class PredictionSentence_Viet_to_HanNom:
    def __init__(self,
                 embedding_size=64,
                 hidden_units=256,
                 max_sentence=30,
                 learning_rate=0.005):

        home = os.getcwd()
        self.max_sentence = max_sentence
        # self.save_dict = home + "/saved_models/{}_vocab.json"
        self.url = 'data/train_dev.csv'

        self.tar_tensor, self.inp_tensor, self.tar_builder, self.inp_builder = DatasetLoader(url=self.url).build_dataset()
        self.values = list(self.tar_builder.index_word.values())
        self.keys = list(self.tar_builder.index_word.keys())

        # Initialize Seq2Seq model
        input_vocab_size = len(self.inp_builder.word_index) + 1
        target_vocab_size = len(self.tar_builder.word_index) + 1

        # input_vocab_size = 3874
        # target_vocab_size = 3874

        # Initialize encoder
        self.encoder = Encode(input_vocab_size,
                              embedding_size,
                              hidden_units)

        # Initialize decoder
        self.decoder = Decoder(target_vocab_size,
                                  embedding_size,
                                  hidden_units)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # Initialize translation
        self.path_save = home + "/saved_models_vn_to_HanNom"
        self.checkpoint_prefix = os.path.join(self.path_save, "Seq2Seq_VN_to_Hannom")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save)).expect_partial()

    def __preprocess_input_text__(self, text):
        inp_lang = remove_punctuation_viet(text)
        vector = [[self.inp_builder.word_index[w] for w in inp_lang.split() if w in list(self.inp_builder.word_index.keys())]]
        sentence = pad_sequences(vector,
                                 maxlen=self.max_sentence,
                                 padding="post",
                                 truncating="post")
        # print(sentence)
        return sentence

    def translate_enroll(self, input_text, label):
        vector = self.__preprocess_input_text__(input_text)
        # Encoder
        encode_outs, last_state = self.encoder(vector)
        # Process decoder input
        input_decode = tf.constant([self.tar_builder.word_index['<sos>']])
        pred_sentence = ""
        for _ in range(self.max_sentence):
            output, last_state = self.decoder(input_decode, encode_outs, last_state)
            # print('output: ',output)
            pred_id = tf.argmax(output, axis=1).numpy() 
            # print('output: ',output)
            # print('pred_id: ',pred_id)
            # print('pred_id type : ',type(pred_id))
            input_decode = pred_id
            word = self.values[pred_id[0]]
            # print(' word : ', word)
            if word not in ["<sos>", "<eos>"]:
                pred_sentence += " " + word
            if word in ["<eos>"]:
                break
        print("-----------------------------------------------------------------")
        print("Input     : ", input_text)
        print("Label     : ", label)
        print("Translate :", pred_sentence)
        print("-----------------------------------------------------------------")

if __name__ == "__main__":
    
    define =  PredictionSentence_HanNom_to_Viet()
    df = pd.read_csv('data/test.csv')
    inp =  df[['Nom','Viet']].sample(10)
    for i, lab in zip(inp['Nom'][:4],inp['Viet'][:4]) :
        define.translate_enroll(i,lab)
    # print('==========nhan===========')
    # print(inp['Viet'][:4])