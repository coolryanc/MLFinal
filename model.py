import os, sys, json, random
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys, os
import parseData

def pair_traing_and_label(trainData, embedding_matrix):
    X_train = []
    y_train = []
    target_y_train = []
    for index in range(len(trainData)-2):
        if index % 1000 == 0:
            print('\rPairing trainData and label, {}'.format(index), end='', flush=True)
        tmp = np.concatenate((trainData[index], trainData[index+1]), axis=0)
        X_train.append(tmp)
        y_train.append(trainData[index+2])
        target_y_sequence = []
        for w in trainData[index+2]:
            target_y_sequence.append(embedding_matrix[w])
        target_y_train.append(target_y_sequence)
    print('')
    return np.array(X_train), np.array(y_train), np.array(target_y_train)

if __name__ == '__main__':
    train_sequence, word_num, sequence_length, embedding_matrix = parseData.gen_sequence()
    X_train, y_train, target_y_train = pair_traing_and_label(train_sequence, embedding_matrix) # (676229, 26), (676229, 13)
    print(X_train.shape)
    print(y_train.shape)
    print(target_y_train.shape)
    VALIDATION_SPLIT = 0.1
    EMBEDDING_DIM = 200
    EPOCH = 20
    BATCH_SIZE = 64
    CELL_SIZE = 256
    decoder_inputs_dim = target_y_train.shape[2]

    num_lstm = np.random.randint(175, 275)
    num_dense = np.random.randint(100, 150)
    rate_drop_lstm = 0.15 + np.random.rand() * 0.25
    rate_drop_dense = 0.15 + np.random.rand() * 0.25
    STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

    # teacher forcing
    decoder_target_data = np.zeros(
        (target_y_train.shape[0], target_y_train.shape[1], target_y_train.shape[2]))
    for index, sentence in enumerate(target_y_train):
        if index % 1000 == 0:
            print('\rTeacher forcing, {} data'.format(index), end='', flush=True)
        for t, word in enumerate(sentence):
            if t < len(sentence)-1:
                decoder_target_data[index][t][:] = target_y_train[index][t+1][:]


    encoder_inputs = Input(shape=(None,))
    x = Embedding(word_num, EMBEDDING_DIM, weights=[embedding_matrix], input_length=sequence_length*2, trainable=False)(encoder_inputs)
    x, state_h, state_c = LSTM(CELL_SIZE,
                           return_state=True)(x)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    x = Embedding(word_num, EMBEDDING_DIM, weights=[embedding_matrix], input_length=sequence_length, trainable=False)(decoder_inputs)
    x = LSTM(CELL_SIZE, return_sequences=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(decoder_inputs_dim, activation='softmax')(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile & run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    model.fit([X_train, y_train], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_split=0.1
              )
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save('./model/s2sEncoder.h5')

    decoder_state_input_h = Input(shape=(CELL_SIZE,))
    decoder_state_input_c = Input(shape=(CELL_SIZE,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()
    decoder_model.save('./model/s2sDecoder.h5')
