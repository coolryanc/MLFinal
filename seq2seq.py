import os, sys, json, random
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.utils import plot_model
import parseData
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim


if __name__ == '__main__':
    w2v = gensim.models.Word2Vec.load('./model/w2v_model_200_10')
    MAX_SEQ_LEN = 15
    EMBEDDING_DIM = 200
    EPOCH = 10
    BATCH_SIZE = 128
    CELL_SIZE = 256

    with open("./data/train_data_seg.txt", "r", encoding = 'utf8') as f:
        all_training = f.read().splitlines()

    t = Tokenizer()
    t.fit_on_texts(all_training)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(all_training)
    padded_docs = pad_sequences(encoded_docs, maxlen=MAX_SEQ_LEN)

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in t.word_index.items():
        if word in w2v:
            embedding_matrix[i] = w2v[word]

    y_train = np.concatenate((padded_docs[1:], padded_docs[0:1]), axis=0)

    # teacher forcing
    decoder_target_data = np.zeros(
        (y_train.shape[0], y_train.shape[1], 1), dtype='float32')
    for index in range(0,len(y_train)-1):
        if index % 1000 == 0:
            print('\rTeacher forcing, {} data'.format(index), end='', flush=True)
        for i in  range(MAX_SEQ_LEN):
            decoder_target_data[index][i] = y_train[index+1][i]

    encoder_inputs = Input(shape=(MAX_SEQ_LEN,))
    x = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False)(encoder_inputs)
    x, state_h, state_c = LSTM(CELL_SIZE,
                           return_state=True)(x)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(MAX_SEQ_LEN,))
    x = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False)(decoder_inputs)
    x = LSTM(CELL_SIZE, return_sequences=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(1, activation='sigmoid')(x)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file='newModel.png', show_shapes=True)
    model.fit([padded_docs, y_train], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=2
              )
    model.save('./model/s2sModelNoneStop.h5') # save model
    # #
    # # #-----------------------------------------------------------------------#
    # encoder_model = Model(encoder_inputs, encoder_states)
    # encoder_model.save('./model/s2sEncoderNoneStop.h5')
    # #
    # decoder_state_input_h = Input(shape=(CELL_SIZE,))
    # decoder_state_input_c = Input(shape=(CELL_SIZE,))
    #
    # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # decoder_outputs, state_h, state_c = decoder_lstm(
    #     decoder_inputs, initial_state=decoder_states_inputs)
    #
    # decoder_states = [state_h, state_c]
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = Model(
    #     [decoder_inputs] + decoder_states_inputs,
    #     [decoder_outputs] + decoder_states)
    # # decoder_model.summary()
    # decoder_model.save('./model/s2sDecoderNoneStop.h5')
    # plot_model(encoder_model, to_file='new_encoder_model.png', show_shapes=True)
    # plot_model(decoder_model, to_file='new_decoder_model.png', show_shapes=True)
