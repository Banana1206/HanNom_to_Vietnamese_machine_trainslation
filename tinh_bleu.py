
import os
import tensorflow as tf
from tqdm import tqdm
from preprocessing import DatasetLoader
from argparse import ArgumentParser
from evaluation import evaluation, evaluation_with_attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Encoder_Attention_decoder import Encode, Attention, Decoder
from metric import Bleu_score
from sklearn.model_selection import train_test_split
import numpy as np
from argparse import ArgumentParser


url = 'data/NomNaNMT.csv'
# url = 'data/train_dev.csv'
url_test = 'data/test.csv'


class bleu_tri:
  def __init__(self,
               embedding_size=64,
               hidden_units = 64,
               learning_rate=0.005,
               test_split_size=0.1,
               epochs = 100,
               batch_size = 128,
               use_bleu=True,
               debug=True
               ):
    self.embedding_size = embedding_size
    self.hidden_units = hidden_units
    self.test_split_size = test_split_size
    self.BATCH_SIZE = batch_size
    self.EPOCHS = epochs
    self.debug = debug
    self.use_bleu = use_bleu

    # save model
    home = os.getcwd()
    self.path_save = home + "/saved_models_HanToViet2"
    if not os.path.exists(self.path_save):
        os.mkdir(self.path_save)

    # Load dataset
    self.inp_tensor, self.tar_tensor, self.inp_builder, self.tar_builder = DatasetLoader(url=url).build_dataset()
    self.inp_tensor_test, self.tar_tensor_test, self.inp_builder_test, self.tar_builder_test = DatasetLoader(url=url_test).build_dataset()

    # Initialize Seq2Seq model
    self.input_vocab_size = len(self.inp_builder.word_index) + 1
    self.target_vocab_size = len(self.tar_builder.word_index) + 1

    # Initialize optimizer
    # if use_lr_schedule:
    #     learning_rate = CustomSchedule(self.hidden_units, warmup_steps=warmup_steps)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Initialize encoder
    self.encoder = Encode(self.input_vocab_size,
                          embedding_size,
                          hidden_units)
    # Initialize decoder with attention
    self.decoder = Decoder(self.target_vocab_size,
                          embedding_size,
                          hidden_units)
    # # Initialize translation
    # self.checkpoint_prefix = os.path.join(self.path_save, "Seq2Seq_hanmon_to_VN")

    # self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
    #                                       encoder=self.encoder,
    #                                       decoder=self.decoder)

     # Initialize translation
    self.path_save = home + "/saved_models_HanToviet4"
    self.checkpoint_prefix = os.path.join(self.path_save, "Seq2Seq_hanmon_to_VN")
    self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                            encoder=self.encoder,
                                            decoder=self.decoder)
    self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save)).expect_partial()

    # Initialize Bleu function
    self.bleu = Bleu_score()

    # train_x, test_x, train_y, test_y = train_test_split(self.inp_tensor, self.tar_tensor, test_size=self.test_split_size)
    # train_x, test_x_, train_y, test_y_ = train_test_split(self.inp_tensor, self.tar_tensor, test_size= 0.)
    # print('len test: ', len(test_x_))
    train_x, test_x, train_y, test_y =  train_test_split(self.inp_tensor_test, self.tar_tensor_test, test_size=0.9)
    print('len train_: ', len(train_x))
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))


    if self.use_bleu:
        print("\n=================================================================")
        
        self.bleu_score = evaluation_with_attention(encoder=self.encoder,
                                                decoder=self.decoder,
                                                test_ds=val_ds,
                                                val_function=self.bleu,
                                                inp_builder=self.inp_builder,
                                                tar_builder=self.tar_builder,
                                                test_split_size=self.test_split_size,
                                                debug=self.debug)

print(bleu_tri().bleu_score)