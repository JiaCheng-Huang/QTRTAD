# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras_preprocessing.text import Tokenizer

import data_helper
import glove_utils
from adversarial_tools import adversarial_paraphrase
from models import *
from unbuffered import Unbuffered

sys.stdout = Unbuffered(sys.stdout)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def fool_text_classifier(model, y_test, test_texts):
    if not os.path.exists('PWWS/fool_result'):
        os.mkdir('PWWS/fool_result')

    classes_prediction = model.predict_classes(test_texts)

    t = Tokenizer(num_words=VOCAB_SIZE)
    t.fit_on_texts(test_texts)

    print('Crafting adversarial examples...')
    model_name = model.__class__.__name__
    adv_text_path = 'PWWS/fool_result/adv_%s.csv' % model_name
    if not os.path.exists(adv_text_path):
        file = open(adv_text_path, "a+", encoding='utf-8')
        file.write('clean' + '\t' + 'adv' + '\t' + 'label' + "\n")
    else:
        file = open(adv_text_path, "r+", encoding='utf-8')
        lines = len(file.readlines())
        test_texts = test_texts[lines - 1:len(test_texts)]
        classes_prediction = classes_prediction[lines - 1:len(classes_prediction)]
        y_test = y_test[lines - 1:len(y_test)]

    for index, text in enumerate(test_texts):
        if y_test[index] == classes_prediction[index]:
            # If the ground_true label is the same as the predicted label
            adv_doc, adv_y, substitute_count = adversarial_paraphrase(input_text=text,
                                                                      true_y=y_test[index],
                                                                      grad_guide=model,
                                                                      tokenizer=t)

            modify_ratio = substitute_count / len(text.split(' '))
            if modify_ratio > 0.25:
                adv_doc = text
                print('High modify ratio:', modify_ratio)
            else:
                print('{}. Adversarial example crafted.'.format(index))
        else:
            adv_doc = text
        file.write(text + '\t' + adv_doc + '\t' + str(y_test[index]) + "\n")
        file.flush()
    file.close()


if __name__ == '__main__':
    seq_length = 200  # 分句长度
    embedding_dim = 50
    VOCAB_SIZE = 50000
    CLAUSE_NUM = 5
    SAMPLES_CAP = 1000

    train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_texts)

    test_texts = test_texts[: SAMPLES_CAP]
    test_labels = test_labels[:SAMPLES_CAP]
    y_test = np.array(test_labels)

    # 加载预训练的GloVe词向量
    glove_model = glove_utils.loadGloveModel('glove.6B.50d.txt')
    embedding_matrix, _ = glove_utils.create_embeddings_matrix(glove_model, tokenizer.word_index, embedding_dim,
                                                               VOCAB_SIZE)
    embedding_matrix = np.transpose(embedding_matrix)
    # model = QNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM)
    # model = TextCNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    # model = BidLSTM(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    model = PlainQNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    # model = Transformer(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    model_name = model.__class__.__name__
    model.load('checkpoint/%s.hdf5' % model_name)
    model.evaluate(test_texts, test_labels)
    fool_text_classifier(model, y_test, test_texts)
