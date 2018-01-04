import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import csv
import gensim
import re
import jieba
import parseData
import sys, os

def main(argv):
    jieba.set_dictionary('extra_dict/dict.txt.big.txt')

    with open("./data/train_data_seg.txt", "r", encoding = 'utf8') as f:
        all_training = f.read().splitlines()

    MAX_SEQ = 15

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_training)

    test_data_path = os.path.join('.', 'data', 'testing_data.csv') # data folder
    testingQuestions, testingAnswers = parseData.read_testing_data(test_data_path)
    test_sequence = tokenizer.texts_to_sequences(testingQuestions)
    test_sequence = pad_sequences(test_sequence, maxlen=MAX_SEQ)

    answer_sequence=[[] for i in range(6)]
    for a in testingAnswers:
        temp = tokenizer.texts_to_sequences(a)
        temp = pad_sequences(temp, maxlen=MAX_SEQ)
        for i in range(6):
            answer_sequence[i].append(temp[i])
    answer_sequence = np.array(answer_sequence)

    model = load_model('./model/Model_400_5.h5')

    predict = [np.zeros((len(testingQuestions),1)) for i in range(6)]
    for m in ['./model/Model_400_5.h5', './model/Model_300_4.h5']:
        model = load_model(m)
        for i in range(6):
            predict[i] += model.predict([test_sequence, answer_sequence[i]])

    with open('./r.csv', 'w') as w:
        w.write('id,ans\n')
        for i in range(len(test_sequence)):
            w.write("{},{}\n".format(i+1, np.argmax([predict[j][i] for j in range(6)])))

if __name__ == '__main__':
    main(sys.argv)
    # pass
