# -*- coding: utf8 -*-
import jieba
import os, re
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
import gensim

def read_training_data(path):
    train_data = []
    for i in range(1,6):
        current_path = os.path.join(path, str(i)+'_train.txt')
        with open(current_path, 'r') as data:
            lines = data.read().splitlines()
            for line in lines:
                train_data.append(line)
    return train_data

def cut(train_data):
    jieba.set_dictionary('extra_dict/dict.txt.big.txt')

    output = open('./data/train_data_seg.txt', 'w')
    for line in train_data:
        temp = ''
        words = jieba.cut(line, cut_all=False)
        for word in words:
            output.write(str(word)+' ')
        output.write('\n')
    output.close()

def main():
    #read train data
    train_data_path = os.path.join('.', 'data', 'training_data')

    train_data = read_training_data(train_data_path)
    #cut sentece to words
    cut(train_data)

    #pretrain W2V
    sentences = word2vec.Text8Corpus('./data/train_data_seg.txt')
    model = word2vec.Word2Vec(sentences, size=200, workers = 6, min_count=10)
    print("vocabulary length: %d"%len(model.wv.vocab))
    model.save("./model/w2v_model_"+str(200)+"_"+str(10))
    model.wv.save_word2vec_format('./data/w2v', binary=False)

if __name__ == '__main__':
    main()
