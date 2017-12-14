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

def pair_traing_and_label(trainData):
    X_train = []
    y_train = []
    for index in range(len(trainData)-2):
        tmp = np.concatenate((trainData[index], trainData[index+1]), axis=0)
        X_train.append(tmp)
        y_train.append(trainData[index+2])
    return np.array(X_train), np.array(y_train)

if __name__ == '__main__':
    train_sequence, word_num, sequence_length, embedding_matrix = parseData.gen_sequence()
    X_train, y_train = pair_traing_and_label(train_sequence)
    print(X_train.shape)
    print(y_train.shape)
    VALIDATION_SPLIT = 0.1

    num_lstm = np.random.randint(175, 275)
    num_dense = np.random.randint(100, 150)
    rate_drop_lstm = 0.15 + np.random.rand() * 0.25
    rate_drop_dense = 0.15 + np.random.rand() * 0.25
    re_weight = True # whether to re-weight classes to fit the 17.5% share in test set
    STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)
	###################
	# build the model #
	###################
	# encoder

    EMBEDDING_DIM = 200
    # u_inputs = Input(shape=(1, 26))
    # u = Embedding(word_num, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=False)(u_inputs)
    # encoder_inputs = dot([u, u_inputs], -1)
    #
    # # encoder_inputs = Input(shape=(200, 26))
    # encoder = LSTM(256, return_state=True)
    # encoder_outputs, encoder_state_h, encoder_state_c = encoder(encoder_inputs)
    # encoder_states = [encoder_state_h, encoder_state_c]
    # # decoder
    # decoder_inputs = Input(shape=(None, word_num))
    # decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # decoder_dense = Dense(word_num, activation='softmax')
    # decoder_outputs = decoder_dense(decoder_outputs)
    # model = Model([u_inputs, decoder_inputs], decoder_outputs)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.summary()

    embedding_layer = Embedding(word_num,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=sequence_length,
        trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    sequence_1_input = Input(shape=(sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation='relu')(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    model = Model(inputs=[X_train, y_train], outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    model.summary()
    print(STAMP)

    # early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    # bst_model_path = STAMP + '.h5'
    # model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    # hist = model.fit([X_train, y_train], labels_train, \
    #     epochs=200, batch_size=64, shuffle=True, \
    #     class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
