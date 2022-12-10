
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

def MaskedSoftmaxCELoss(label, pred):
    """
    :param label: shape (batch_size, max_length, vocab_size)
    :param pred: shape (batch_size, max_length)
    :return: weighted_loss: shape (batch_size, max_length)
    """
    weights_mask = 1 - np.equal(label, 0)
    unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, pred)
    weighted_loss = tf.reduce_mean(unweighted_loss * weights_mask)
    return weighted_loss

class Seq2Seq:
  def __init__(self,
               embedding_size=64,
               hidden_units = 256,
               learning_rate=0.005,
               test_split_size=0.1,
               epochs = 10,
               batch_size = 128,
               use_bleu=False,
               debug=False
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
    self.path_save = home + "/saved_models"
    if not os.path.exists(self.path_save):
        os.mkdir(self.path_save)

    # Load dataset
    self.inp_tensor, self.tar_tensor, self.inp_builder, self.tar_builder = DatasetLoader(url=url).build_dataset()

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
    # Initialize translation
    self.checkpoint_prefix = os.path.join(self.path_save, "Seq2Seq_hanmon_to_VN")

    self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                          encoder=self.encoder,
                                          decoder=self.decoder)
    # Initialize Bleu function
    self.bleu = Bleu_score()

  # def train_step(self, x, y):
  #   with tf.GradientTape() as tape:
  #     # Teacher forcing
  #     sos = tf.reshape(tf.constant([self.tar_builder.word_index['<sos>']] * self.BATCH_SIZE),
  #                       shape=(-1, 1))
  #     dec_input = tf.concat([sos, y[:, :-1]], 1)
  #     # Encoder
  #     _, last_state = self.encoder(x)
  #     # Decoder
  #     outs, last_state = self.decoder(dec_input, last_state)
  #     # Loss
  #     loss = MaskedSoftmaxCELoss(y, outs)

  #   train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
  #   grads = tape.gradient(loss, train_vars)
  #   self.optimizer.apply_gradients(zip(grads, train_vars))
  #   return loss

  def train_step_with_attention(self, x, y):
    loss = 0
    with tf.GradientTape() as tape:
        # Teaching force
        dec_input = tf.constant([self.tar_builder.word_index['<sos>']] * self.BATCH_SIZE)
        # Encoder
        encoder_outs, last_state = self.encoder(x)
        for i in range(1, y.shape[1]):
            # Decoder
            decode_out, last_state = self.decoder(dec_input, encoder_outs, last_state)
            # Loss
            loss += MaskedSoftmaxCELoss(y[:, i], decode_out)
            # Decoder input
            dec_input = y[:, i]

    train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
    grads = tape.gradient(loss, train_vars)
    self.optimizer.apply_gradients(zip(grads, train_vars))
    return loss

  def training(self):
    # Add to tensor
    train_x, test_x, train_y, test_y = train_test_split(self.inp_tensor, self.tar_tensor, test_size=self.test_split_size)

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    N_BATCH = train_x.shape[0] // self.BATCH_SIZE

    tmp = 0
    for epoch in range(self.EPOCHS):
        total_loss = 0
        print( "( {},{} )".format(epoch,self.EPOCHS))
        for _, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
          total_loss += self.train_step_with_attention(x, y)
      
        if self.use_bleu:
          print("\n=================================================================")
          
          bleu_score = evaluation_with_attention(encoder=self.encoder,
                                                  decoder=self.decoder,
                                                  test_ds=val_ds,
                                                  val_function=self.bleu,
                                                  inp_builder=self.inp_builder,
                                                  tar_builder=self.tar_builder,
                                                  test_split_size=self.test_split_size,
                                                  debug=self.debug)
          print("-----------------------------------------------------------------")
          print(f'Epoch {epoch + 1}/{self.EPOCHS} -- Loss: {total_loss} -- Bleu_score: {bleu_score}')
          if bleu_score > tmp:
              self.checkpoint.save(file_prefix=self.checkpoint_prefix)
              print("[INFO] Saved model in '{}' direction!".format(self.path_save))
              tmp = bleu_score
          print("=================================================================\n")
        else:
          print("=================================================================")
          print(f'Epoch {epoch + 1}/{self.EPOCHS} -- Loss: {total_loss}')
          print("=================================================================\n")
    
    print("[INFO] Saved model in '{}' direction!".format(self.path_save))
    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

if __name__ == "__main__":

  Seq2Seq().training()
  # pass