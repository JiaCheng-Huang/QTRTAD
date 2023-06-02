# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import glove_utils
import os
import sys

from models.QNN_backup import QNN

import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer

import data_helper
from adversarial_tools import adversarial_paraphrase
from models.TextCNN import TextCNN
from unbuffered import Unbuffered
from keras import backend as K

sys.stdout = Unbuffered(sys.stdout)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def write_origin_input_texts(origin_input_texts_path, test_texts, test_samples_cap=None):
    if test_samples_cap is None:
        test_samples_cap = len(test_texts)
    with open(origin_input_texts_path, 'a') as f:
        for i in range(test_samples_cap):
            f.write(test_texts[i] + '\n')


def fool_text_classifier(model, y_test, test_texts):
    if not os.path.exists('PWWS/fool_result'):
        os.mkdir('PWWS/fool_result')
    clean_texts_path = 'clean.txt'
    if not os.path.isfile(clean_texts_path):
        write_origin_input_texts(clean_texts_path, test_texts)

    classes_prediction = model.predict_classes(test_texts)

    t = Tokenizer(num_words=VOCAB_SIZE)
    t.fit_on_texts(test_texts)

    print('Crafting adversarial examples...')
    model_name = model.__class__.__name__
    adv_text_path = 'PWWS/fool_result/adv_%s.txt' % model_name
    file = open(adv_text_path, "a",encoding='utf-8')
    for index, text in enumerate(test_texts):
        if np.argmax(y_test[index]) == classes_prediction[index]:
            # If the ground_true label is the same as the predicted label
            adv_doc, adv_y = adversarial_paraphrase(input_text=text,
                                                    true_y=np.argmax(
                                                        y_test[index]),
                                                    grad_guide=model,
                                                    tokenizer=t,
                                                    dataset='imdb')
            print('{}. Adversarial example crafted.'.format(index))

            text = adv_doc
        file.write(text + "\n")
        file.flush()
    file.close()


if __name__ == '__main__':
    seq_length = 200  # 分句长度
    embedding_dim = 50
    VOCAB_SIZE = 50000
    CLAUSE_NUM = 2
    SAMPLES_CAP = 1000

    train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_texts)

    test_texts = test_texts[: SAMPLES_CAP]
    y_test = np.array(test_labels)

    # 加载预训练的GloVe词向量
    glove_model = glove_utils.loadGloveModel('glove.6B.50d.txt')
    embedding_matrix, _ = glove_utils.create_embeddings_matrix(glove_model, tokenizer.word_index, embedding_dim,
                                                               VOCAB_SIZE)
    embedding_matrix = np.transpose(embedding_matrix)
    # model = QTRTnn(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM)
    # model = QNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM)
    model = TextCNN(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)
    model.load('checkpoint/cnn.hdf5')
    model.evaluate(test_texts, test_labels[:SAMPLES_CAP])
    fool_text_classifier(model, y_test, test_texts)
