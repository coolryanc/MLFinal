import jieba
import os, re
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer

def read_training_data(path):
    train_data = []
    for i in range(1,6):
        current_path = os.path.join(path, str(i)+'_train.txt')
        with open(current_path, 'r') as data:
            lines = data.read().splitlines()
            for line in lines:
                train_data.append(line)
    return train_data

def cut_training_data(train_data):
    train_text = []
    for line in train_data:
        tmp = []
        words = jieba.cut(line, cut_all=False)
        for word in words:
            tmp.append(word)
        train_text.append(tmp)
    return train_text



if __name__ == '__main__':
    train_data = read_training_data('./provideData/training_data')
    train_data = cut_training_data(train_data)
    print(train_data[:10])
