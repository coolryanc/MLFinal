import sys
import os, re
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import random
import json
import numpy as np

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
jieba.set_dictionary('extra_dict/dict.txt.big.txt')

def cut(train_data):
    train_text = []
    for line in train_data:
        temp = ''
        words = jieba.cut(line, cut_all=False)
        for word in words:
            temp += str(word) + ' '
        temp = temp.strip()
        train_text.append(temp)
    with open('trainData.txt', 'w') as outfile:
        json.dump(train_text, outfile)

def cut_sentence(_str):
    s = ''
    words = jieba.cut(_str, cut_all=False)
    for word in words:
        s += str(word) + ' '
    s = s.strip()
    return s

def replace_string(_str):
    s = re.sub(r"A:", r"", _str)
    s = re.sub(r"B:", r"", s)
    s = re.sub(r"A", r"", s)
    s = re.sub(r"B", r"", s)
    s = removeTwoWords(s)
    return s

def removeTwoWords(_str):
    s = _str.split()
    sentence = [i for i in s if len(i) > 2]
    sentence = "".join(sentence)
    sentence = re.sub(r"\s+","", sentence)
    return sentence

def read_testing_data(path):
    testingQuestions = []
    testingAnswers = []
    with open(path, 'r') as data:
        lines = data.read().splitlines()[1:]
        for index, line in enumerate(lines):
            line = line.replace('Ａ','A')
            line = line.replace('Ｂ','B')
            sentences = line.split(',')[1:]
            q = replace_string(sentences[0])
            answers = sentences[1].split('\t') # split answer to list
            answers = [cut_sentence(removeTwoWords(re.sub(r"[A-Z]:", r"", i))) for i in answers]
            testingQuestions.append(cut_sentence(q))
            testingAnswers.append(answers)
    return testingQuestions, testingAnswers

def pad_testing_sequence(testingQuestions, testingAnswers):
    training_data = json.load(open('./trainData.txt'))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_data)
    word_num = len(tokenizer.word_index) + 1 # padding
    test_sequence = tokenizer.texts_to_sequences(testingQuestions)
    test_sequence = pad_sequences(test_sequence, maxlen=26)
    answer_sequence=[]
    for a in testingAnswers:
        temp = tokenizer.texts_to_sequences(a)
        temp = pad_sequences(temp, maxlen=13)
        answer_sequence.append(temp)
    return test_sequence, answer_sequence

def read_training_data(path):
    training_data = []
    for dirPath, dirNames, fileNames in os.walk(path):
        for f in fileNames:
            filePath = os.path.join(dirPath, f)
            with open(filePath, 'r') as readFile:
                lines = readFile.read().splitlines()
                for line in lines:
                    no_punct = ""
                    for char in line:
                       if char not in punctuations:
                           no_punct = no_punct + char
                    if len(no_punct) > 2:
                        training_data.append(no_punct)
    return training_data

def gen_sequence():
    training_data = json.load(open('./trainData.txt'))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_data)
    word_num = len(tokenizer.word_index) + 1 # padding
    train_sequence = tokenizer.texts_to_sequences(training_data)
    train_sequence = pad_sequences(train_sequence)
    sequence_length = train_sequence.shape[1]
    word_index = tokenizer.word_index

    embedding_index = {}
    with open('w2v','r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_index[word] = coefs
    print ('Found %s word vectors.' % len(embedding_index))

    EMBEDDING_DIM = 200
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    np.save('embeddingMatrix', embedding_matrix)
    return train_sequence, word_num, sequence_length, embedding_matrix

if __name__ == '__main__':
    test_data_path = os.path.join('.', 'data', 'testing_data.csv') # data folder
    train_data_path = os.path.join('.', 'data', 'training_data') # data folder
    # testingQuestions, testingAnswers = read_testing_data(test_data_path)
    # training_data = read_training_data(train_data_path)
    # cut(training_data)
    gen_sequence()
