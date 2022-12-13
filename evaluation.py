import numpy as np
import collections
import tensorflow as tf


def evaluation(encoder,
               decoder,
               test_ds,
               val_function,
               inp_builder,
               tar_builder,
               test_split_size,
               debug=False):
    """
    :param test_ds: (inp_vocab, tar_vocab)
    :param (inp_lang, tar_lang)
    :return:
    """
    # Preprocessing testing data
    score = 0.0
    count = 0
    test_ds_len = int(len(test_ds) * test_split_size)
    for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1).take(test_ds_len):
        test_x = tf.expand_dims(test_, axis=0)
        _, last_state = encoder(test_x)

        input_decode = tf.reshape(tf.constant([tar_builder.word_index['<sos>']]), shape=(-1, 1))
        sentence = []
        for _ in range(len(test_y)):
            output, last_state = decoder(input_decode, last_state, training=False)
            output = tf.argmax(output, axis=2).numpy()
            input_decode = output
            sentence.append(output[0][0])

        input_sequence = inp_builder.sequences_to_texts([test_.numpy()])[0]
        pred_sequence = tar_builder.sequences_to_texts([sentence])[0]
        target_sequence = tar_builder.sequences_to_texts([test_y.numpy()])[0]
        score += val_function(pred_sequence,
                              target_sequence)
        if debug and count <= 5:
            print("-----------------------------------------------------------------")
            print("Input    : ", input_sequence)
            print("Predicted: ", pred_sequence)
            print("Target   : ", target_sequence)
            count += 1
    return score / test_ds_len


def evaluation_with_attention(encoder,
                              decoder,
                              test_ds,
                              val_function,
                              inp_builder,
                              tar_builder,
                              test_split_size,
                              debug=False):
    """
    :param test_ds: (inp_vocab, tar_vocab)
    :param (inp_lang, tar_lang)
    :return:
    """
    # Preprocessing testing data
    score = 0.0
    count = 0
    test_ds_len = int(len(test_ds) * test_split_size)
    for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1).take(test_ds_len):
        test_x = tf.expand_dims(test_, axis=0)
        encode_outs, last_state = encoder(test_x)
        input_decode = tf.constant([tar_builder.word_index['<sos>']])
        sentence = []
        for _ in range(len(test_y)):
            output, last_state = decoder(input_decode, encode_outs, last_state, training=False)
            pred_id = tf.argmax(output, axis=1).numpy()
            input_decode = pred_id
            sentence.append(pred_id[0])

        input_sequence = inp_builder.sequences_to_texts([test_.numpy()])[0]
        pred_sequence = tar_builder.sequences_to_texts([sentence])[0]
        target_sequence = tar_builder.sequences_to_texts([test_y.numpy()])[0]

        score += val_function(pred_sequence,
                              target_sequence)
        if debug and count <= 5:
            print("-----------------------------------------------------------------")
            print("Input    : ", input_sequence)
            print("Predicted: ", pred_sequence)
            print("Target   : ", target_sequence)
            count += 1
    return score / test_ds_len