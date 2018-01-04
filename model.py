import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Activation, Input, Embedding, LSTM, Reshape, Bidirectional, GRU
from keras.models import Model
from keras.layers.merge import dot

import pandas as pd
import numpy as np
import csv
import gensim
import re



if __name__ == '__main__':
    w2v = gensim.models.Word2Vec.load('./model/w2v_model_200_10')
    MAX_SEQ_LEN = 15
    EMBEDDING_DIM = 200

    with open("./data/train_data_seg.txt", "r", encoding = 'utf8') as f:
        all_training = f.read().splitlines()

    t = Tokenizer()
    t.fit_on_texts(all_training)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(all_training)
    padded_docs = pad_sequences(encoded_docs, maxlen=MAX_SEQ_LEN)

    duplicate_num = 4 #4
    data_len = len(padded_docs)
    shift_bias = 250 #300

    y_train = np.concatenate((np.ones(data_len), np.zeros(data_len)), axis=0)
    for i in range(duplicate_num - 1):
        y_train = np.concatenate((y_train, np.zeros(data_len)), axis=0)

    tmp = np.concatenate((padded_docs, padded_docs), axis=0)
    for i in range(duplicate_num - 1):
        tmp = np.concatenate((tmp, padded_docs), axis=0)
    training_seq_one = tmp

    tmp = np.concatenate((padded_docs[1:], padded_docs[0:1]), axis=0)

    tmp = np.concatenate((tmp, padded_docs[int(data_len/2):]), axis=0)
    tmp = np.concatenate((tmp, padded_docs[0:int(data_len/2)]), axis=0)
    for i in range(duplicate_num - 1):
        tmp = np.concatenate((tmp, padded_docs[(int(data_len/2)+(i+1)*shift_bias):]), axis=0)
        tmp = np.concatenate((tmp, padded_docs[0:(int(data_len/2)+(i+1)*shift_bias)]), axis=0)
    training_seq_two = tmp

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    print(embedding_matrix.shape)
    for word, i in t.word_index.items():
        if word in w2v:
            embedding_matrix[i] = w2v[word]

    input_seq_one = Input(shape=(MAX_SEQ_LEN,))
    net_seq_one = input_seq_one
    net_seq_one = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False)(net_seq_one)

    input_seq_two = Input(shape=(MAX_SEQ_LEN,))
    net_seq_two = input_seq_two
    net_seq_two = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False)(net_seq_two)

    net_seq_one = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.2))(net_seq_one)
    net_seq_two = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.2))(net_seq_two)

    dot_product = dot([net_seq_one, net_seq_two], axes= 1)
    output = Dense(1, activation='sigmoid')(dot_product)


    model = Model(inputs=[input_seq_one, input_seq_two], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot_model(model, to_file='Model.png', show_shapes=True)
    print(model.summary())
    model.fit([training_seq_one, training_seq_two], y_train, epochs=8, batch_size=128)
    model.save('./model/Model_250_4.h5')
