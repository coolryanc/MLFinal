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

def read_testing_data(path):
    testing_data = []
    with open(path, 'r') as data:
        lines = data.read().splitlines()[1:]
        for line in lines:
            sentences = line.split(',')[1:]
            for sentence in sentences:
                sentence = re.sub(r"A:", r"", sentence)
                sentence = re.sub(r"B:", r"", sentence)
                sentence = re.sub(r"，", r"", sentence)
                sentence = re.sub(r"...", r"", sentence)
                tmp = sentence.split()
                for item in tmp:
                    testing_data.append(item)
    return testing_data

def cut(train_data):
    jieba.set_dictionary('extra_dict/dict.txt.big')

    output = open('cut_data.txt', 'w')
    train_text = []
    for line in train_data:
        temp = ''
        words = jieba.cut(line, cut_all=False)
        for word in words:
            output.write(str(word)+' ')
        output.write('\n')
        train_text.append(temp)
    output.close()

def main():

    #read train data
    test_data_path = os.path.join('.', 'provideData', 'testing_data.csv')
    train_data_path = os.path.join('.', 'provideData', 'training_data')

    test_data = read_testing_data(test_data_path)
    train_data = read_training_data(train_data_path)

    train_data = train_data + test_data
    #cut sentece to words
    cut(train_data)

    #pretrain W2V
    sentences = word2vec.Text8Corpus('cut_data.txt')
    model = word2vec.Word2Vec(sentences, size=200)
    model.wv.save_word2vec_format('w2v', binary=False)

if __name__ == '__main__':
    main()
