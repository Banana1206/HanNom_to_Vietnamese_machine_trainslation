import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, LSTM,  Dense

class Encode(tf.keras.Model):
  def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
    """
            Encoder block in Sequence to Sequence
        :param vocab_size: Số lượng từ của bộ từ vựng đầu vào
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
    """ 
    super(Encode, self).__init__(**kwargs)

    self.hidden_units = hidden_units

    # The embedding layer converts tokens to vectors
    self.embedding = Embedding(vocab_size, embedding_size)

    # The RNN layer processes those vectors sequentially
    self.encode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
  def __call__(self, x, *args, **kwargs):
    """
        :Input:
            - x: [batch_size, max_length]
        :return:
            - output: [batch_size, embedding_dim, Hidden_unites]
            - state_h: [batch_size, hidden_units] - Current Hidden state
            - state_c: [batch_size, hidden_units] - Current Cell state
    """
    first_state = self.init_hidden_state(x.shape[0])
    encode = self.embedding(x)
    encode, state_h, state_c = self.encode_layer_1(encode, first_state, **kwargs)
    return encode, [state_h, state_c]

  def init_hidden_state(self, batch_size):
    return [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]


class Attention(Layer):

    def __init__(self, hidden_units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.Wa = Dense(hidden_units)

    def __call__(self, encoder_outs, decoder_outs, *args, **kwargs):
        score = tf.matmul(decoder_outs, self.Wa(encoder_outs), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=2)
        context_vector = tf.matmul(alignment, encoder_outs)
        return context_vector, score

class Decoder(tf.keras.Model):
    """
        Luong Attention layer in Seq2Seq: https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        # 1. The embedding layer converts token IDs to vectors
        self.embedding = Embedding(vocab_size, embedding_size)

        # 2. The RNN keeps track of what's been generated so far.
        self.decode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        
        # 3. The RNN output will be the query for the attention layer.
        self.attention = Attention(hidden_units=hidden_units)

        # 4. This fully connected layer produces the logits for each
        # output token.
        self.dense = Dense(vocab_size)

    def __call__(self, x, encoder_outs, state, *args, **kwargs):
        """
        :Input:
            - x: [batch_size, max_length]
            - encode_output: [batch_size, max_length, hidden_units]
            - State:
                + state_h: [batch_size, hidden_units] - Hidden state in encode layer
                + state_c: [batch_size, hidden_units] - Cell state in encode layer
        :return:
            - output: [batch_size, vocab_size]
            - state_h: [batch_size, hidden_units] - Current Hidden state
            - state_c: [batch_size, hidden_units] - Current Cell state
        """
        x = tf.expand_dims(x, axis=1)
        x = self.embedding(x)
        decode_outs, state_h, state_c = self.decode_layer_1(x, state)
        context_vector, att_weights = self.attention(encoder_outs, decode_outs)
        concat = tf.concat([decode_outs, context_vector], axis=-1)
        concat = tf.reshape(concat, (-1, concat.shape[2]))
        outs = self.dense(concat)
        return outs, [state_h, state_c]
