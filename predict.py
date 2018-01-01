import os, sys, json, random
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import parseData
from scipy import spatial
from gensim.models import word2vec
import gensim

embedding_matrix = np.load('embeddingMatrix.npy')
index2Word = json.load(open('index2Word.txt'))
embeddModel = gensim.models.Word2Vec.load("./model/ML1.w2v")

def seq2Vector(li):
    y = []
    for i in li:
        y.append(embedding_matrix[i])
    return np.array(y)

def predict_sequence(encoder_model, decoder_model, input_seq, answer_seq, n_steps=9, cardinality = 200):
    v1 = seq2Vector(input_seq)
    state = encoder_model.predict(v1)

    target_seq = np.zeros((1, 1, cardinality))
    output = list()
    for t in range(n_steps):
        y, h, c = decoder_model.predict([target_seq] + state)
        output.append(y[0][0])
        state = [h, c]
        target_seq = y
    output = np.array(output)
    _m = sys.maxsize
    ans = 0
    for index, seq in enumerate(answer_seq[0]):
        v = seq2Vector(seq)
        dis = np.linalg.norm(output - v)
        print(dis)
        if dis < _m:
            _m = dis
            ans = index
    print('=====================')
    return ans
    # for index, seq in enumerate(answer_seq[0]):
    #     tempSim = 0
    #     for j, s in enumerate(seq):
    #         if s != 0:
    #             print('{} {}'.format(s, index2Word[str(s)]), end='')
    #         if str(s) in index2Word and str(output[j][0]) in index2Word:
    #             # print(index2Word[str(s)], end='')
    #             sim = model.wv.similarity(index2Word[str(s)], index2Word[str(output[j][0])])
    #             if sim > 0.3:
    #                 tempSim += sim

def main(argv):
    test_data_path = os.path.join('.', 'data', 'testing_data.csv') # data folder
    rawTestingQuestions, rawTestingAnswers = parseData.read_testing_data(test_data_path)
    test_sequence, answer_sequence = parseData.pad_testing_sequence(rawTestingQuestions, rawTestingAnswers)
    encoder_model = load_model('./model/s2sEncoderNoneStop.h5')
    decoder_model = load_model('./model/s2sDecoderNoneStop.h5')
    fd = open('result1229.csv','a')
    fd.write('id,ans\n')
    fd.close()
    for seq_index in range(10):
        # print('Input sentence: {}'.format("".join(rawTestingQuestions[seq_index])))
        input_seq = test_sequence[seq_index: seq_index + 1]
        answer_seq = answer_sequence[seq_index: seq_index + 1]
        ans = predict_sequence(encoder_model, decoder_model, input_seq, answer_seq)
        print('-')
        print('Input sentence: {} \nANSWER: {} {}'.format("".join(rawTestingQuestions[seq_index]), ans, "".join(rawTestingAnswers[seq_index][ans])))
        fd = open('result1229.csv','a')
        fd.write('{},{}\n'.format(seq_index+1, ans))
        fd.close()

if __name__ == '__main__':
    main(sys.argv)
    # pass
