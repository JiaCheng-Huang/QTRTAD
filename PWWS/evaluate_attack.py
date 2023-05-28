import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer

import data_helper
import glove_utils
from models.QNN import QNN
from models.QTRT import QTRTnn
from models.TextCNN import TextCNN

if __name__ == '__main__':
    seq_length = 200  # 分句长度
    embedding_dim = 50
    VOCAB_SIZE = 50000
    CLAUSE_NUM = 5
    SAMPLES_CAP = 164

    train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_texts + test_texts)

    glove_model = glove_utils.loadGloveModel('glove.6B.50d.txt')
    embedding_matrix, _ = glove_utils.create_embeddings_matrix(glove_model, tokenizer.word_index, embedding_dim,
                                                               VOCAB_SIZE)
    embedding_matrix = np.transpose(embedding_matrix)
    # model = QTRTnn(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM)
    # model = QNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM)
    # model.load('qnn.hdf5')

    model = TextCNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    model.load('cnn.hdf5')

    model.evaluate(test_texts[:SAMPLES_CAP], test_labels[:SAMPLES_CAP])
    adv = []
    with open('fool_result/adv_QNN.txt') as f:
        for line in f.readlines():
            adv.append(line)
    model.evaluate(adv, test_labels[:SAMPLES_CAP])
