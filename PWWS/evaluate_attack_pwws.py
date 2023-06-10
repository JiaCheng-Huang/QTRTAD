import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer

import data_helper
import glove_utils
from models import *

if __name__ == '__main__':
    seq_length = 200  # 分句长度
    embedding_dim = 50
    VOCAB_SIZE = 50000
    CLAUSE_NUM = 5
    SAMPLES_CAP = 1000

    train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_texts)

    glove_model = glove_utils.loadGloveModel('glove.6B.50d.txt')
    embedding_matrix, _ = glove_utils.create_embeddings_matrix(glove_model, tokenizer.word_index, embedding_dim,
                                                               VOCAB_SIZE)
    embedding_matrix = np.transpose(embedding_matrix)
    # model = QNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM)
    model = PlainQNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    # model = TextCNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    # model = BidLSTM(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    # model = Transformer(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    model_name = model.__class__.__name__
    model.load('checkpoint/%s.hdf5' % model_name.lower())
    df = pd.read_csv('PWWS/fool_result/adv_%s.csv' % model_name, sep='\t')
    label = df['label']
    clean = df['clean']
    adv = df['adv']

    model.evaluate(clean.tolist(), label.tolist())

    model.evaluate(adv.tolist(), label.tolist())

    # with open('PWWS/fool_result/adv_TextCNN.csv') as f:
    #     lines = f.readlines()
    #     for i in range(1, len(lines)):
    #         adv.append(lines[i].split('\t')[1])
    #         adv_label.append(lines[i].split('\t')[2].replace('\n', ''))
    # model.evaluate(adv, adv_label)
