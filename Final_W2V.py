import jieba
import os
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

def cut(train_data):
    jieba.set_dictionary('extra_dict/dict.txt.big')

    output = open('cut_data.txt', 'w')
    train_text = []
    for line in train_data:
        temp = ''
        words = jieba.cut(line, cut_all=False)
        for word in words:
            output.write(str(word))
        output.write('\n')
        train_text.append(temp)
    output.close()

def main():

    #read train data
    data_path = os.path.join('.', 'provideData', 'training_data')
    train_data = read_training_data(data_path)

    #cut sentece to words
    cut(train_data)

    #pretrain W2V
    sentences = word2vec.Text8Corpus('cut_data.txt')
    model = word2vec.Word2Vec(sentences, size=200)
    model.wv.save_word2vec_format('w2v', binary=False)

if __name__ == '__main__':
    main()
