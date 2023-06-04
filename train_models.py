# -*- coding: utf-8 -*-
import nltk
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.text import Tokenizer

import data_helper
import glove_utils
from models.QNN import QNN
from models.BidLSTM import BidLSTM
from models.TextCNN import TextCNN
from models.Transformer import Transformer

# nltk.data.path.append("nltk_data")

if __name__ == '__main__':
    seq_length = 200
    embedding_dim = 50
    VOCAB_SIZE = 50000
    CLAUSE_NUM = 5

    train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_texts)

    # 加载预训练的GloVe词向量
    glove_model = glove_utils.loadGloveModel('glove.6B.50d.txt')
    embedding_matrix, _ = glove_utils.create_embeddings_matrix(glove_model, tokenizer.word_index, embedding_dim,
                                                               VOCAB_SIZE)
    embedding_matrix = np.transpose(embedding_matrix)
    # model = BidLSTM(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    # model = TextCNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    model = Transformer(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    # model = QNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM)
    model.train(train_texts, train_labels, batch_size=16, epochs=80,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                           ReduceLROnPlateau(monitor='val_loss', patience=2)]
                , validation_split=0.2)
    scores = model.evaluate(test_texts, test_labels)
    model.save('checkpoint/transformer.hdf5')
