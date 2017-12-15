import os, sys, json, random
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import parseData

embedding_matrix = np.load('embeddingMatrix.npy')

def seq2Vector(li):
    y = []
    for i in li:
        y.append(embedding_matrix[i])
    return np.array(y)

def predict_sequence(encoder_model, decoder_model, input_seq, answer_seq, n_steps=13, cardinality = 200):
    state = encoder_model.predict(input_seq)

    target_seq = np.array([0.0 for _ in range(13)]).reshape(1, 13)
    y, h, c = decoder_model.predict([target_seq] + state)
    y = np.array(y)
    _m = sys.maxsize
    ans = 0
    for index, seq in enumerate(answer_seq[0]):
        v = seq2Vector(seq)
        dis = np.linalg.norm(y - v)
        if dis < _m:
            _m = dis
            ans = index
    return ans

def main(argv):
    test_data_path = os.path.join('.', 'data', 'testing_data.csv') # data folder
    rawTestingQuestions, rawTestingAnswers = parseData.read_testing_data(test_data_path)
    test_sequence, answer_sequence = parseData.pad_testing_sequence(rawTestingQuestions, rawTestingAnswers)
    encoder_model = load_model('./model/s2sEncoder.h5')
    decoder_model = load_model('./model/s2sDecoder.h5')
    writeText = 'id,ans\n'
    for seq_index in range(len(rawTestingQuestions)):
        # Take one sequence (part of the training test)
        # for trying out decoding.
        input_seq = test_sequence[seq_index: seq_index + 1]
        answer_seq = answer_sequence[seq_index: seq_index + 1]
        ans = predict_sequence(encoder_model, decoder_model, input_seq, answer_seq)
        print('-')
        print('Input sentence: {} \nANSWER: {} {}'.format("".join(rawTestingQuestions[seq_index]), ans, "".join(rawTestingAnswers[seq_index][ans])))
        fd = open('result.csv','a')
        fd.write('{},{}\n'.format(seq_index+1, ans))
        fd.close()

if __name__ == '__main__':
    main(sys.argv)
