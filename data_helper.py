import numpy as np
import re
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.contrib import learn

class word_vector(object):

    def __init__(self, file_path, max_sentence_length=4):
        self.file_path = file_path

        with open(self.file_path, 'r') as f:
            text = f.readlines()

        self.words_dict, self.data = {}, []
        for i, j in enumerate(map(lambda x: x.split(' '), text)):
            self.words_dict[j[0]] = i + 1
            self.data.append(list(map(float, j[1:])))

        # add an index for words that are not in word set.
        self.vocab_size = len(self.words_dict) + 1
        self.words_dict['<unk>'] = 0

        # add a row of 0 for words that are not in word set.
        self.embedding_size = len(self.data[0])
        # self.data.append([0]*self.embedding_size)
        self.data = np.array(self.data)

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_sentence_length, tokenizer_fn=self.tokenize, vocabulary=self.words_dict)

    def tokenize(self, iterator):
        for i in iterator:
            lis = []
            for j in i.split(' '):
                if j not in self.vocab_processor.vocabulary_:
                    j = "<unk>"
                lis.append(j)
            yield lis


# function to clean string
def clean_str(string):
    '''
    Tokenization/string cleaning forn all dataset except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# function to load data: both covariant and variant:
def load_data(file_path):
    da = pd.read_table(file_path, sep='\t')

    # extract variant
    y = np.array(da['relatedness_score']).reshape((-1, 1))

    # extract sentence A
    sentence_A = [clean_str(i) for i in da['sentence_A']]
    sentence_B = [clean_str(i) for i in da['sentence_B']]
    return sentence_A, sentence_B, y

# function to generate batches
def batches_generate(data, epoch_size, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    There will be epoch_size * num_batch_per_epoch batches in total
    """

    data = np.array(data)

    # records of data
    data_size = len(data)

    # batches per epoch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(epoch_size):
        # Shuffle the data ata each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]
