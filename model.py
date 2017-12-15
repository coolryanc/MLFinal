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
from keras.utils import plot_model
import parseData

def pair_traing_and_label(trainData, embedding_matrix):
    X_train = []
    target_y_train = []
    for index in range(len(trainData)-1):
        if index % 1000 == 0:
            print('\rPairing trainData and label, {}'.format(index), end='', flush=True)
        # tmp = np.concatenate((trainData[index], trainData[index+1]), axis=0)
        target_X_sequence = []
        for w in trainData[index]:
            target_X_sequence.append(embedding_matrix[w])
        target_y_sequence = []
        for w in trainData[index+1]:
            target_y_sequence.append(embedding_matrix[w])

        X_train.append(target_X_sequence)
        target_y_train.append(target_y_sequence)
    print('')
    return np.array(X_train), np.array(target_y_train)

if __name__ == '__main__':
    train_sequence, word_num, sequence_length, embedding_matrix = parseData.gen_sequence()
    X_train, y_train = pair_traing_and_label(train_sequence, embedding_matrix) # (676229, 26), (676229, 13)
    print(X_train.shape)
    print(y_train.shape)
    VALIDATION_SPLIT = 0.1
    EMBEDDING_DIM = 200
    EPOCH = 16
    BATCH_SIZE = 32
    CELL_SIZE = 256
    decoder_inputs_dim = y_train.shape[2]

    num_lstm = np.random.randint(175, 275)
    num_dense = np.random.randint(100, 150)
    rate_drop_lstm = 0.15 + np.random.rand() * 0.25
    rate_drop_dense = 0.15 + np.random.rand() * 0.25
    STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

    # teacher forcing
    decoder_target_data = np.zeros(
        (y_train.shape[0], y_train.shape[1], y_train.shape[2]), dtype='float32')
    for index, sentence in enumerate(y_train):
        if index % 1000 == 0:
            print('\rTeacher forcing, {} data'.format(index), end='', flush=True)
        for t, word in enumerate(sentence):
            if t < len(sentence)-1:
                decoder_target_data[index][t][:] = y_train[index][t+1][:]

    encoder_inputs = Input(shape=(None, EMBEDDING_DIM))
    encoder = LSTM(CELL_SIZE, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, decoder_inputs_dim))
    decoder_lstm = LSTM(CELL_SIZE, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(decoder_inputs_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    earlystopping = EarlyStopping(monitor='val_loss', patience = 10)
    checkpoint = ModelCheckpoint(filepath='./model/currentBest.h5', save_best_only=True, monitor='val_loss')
    model.summary()
    plot_model(model, to_file='newModel.png', show_shapes=True)
    model.fit([X_train[:128], y_train[:128]], decoder_target_data[:128],
              batch_size=64,
              epochs=1,
              # callbacks=[earlystopping, checkpoint]
              )
    model.save('./model/s2sModel.h5') # save model

    #-----------------------------------------------------------------------#
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save('./model/s2sEncoder.h5')
    #
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
    plot_model(encoder_model, to_file='new_encoder_model.png', show_shapes=True)
    plot_model(decoder_model, to_file='new_decoder_model.png', show_shapes=True)
