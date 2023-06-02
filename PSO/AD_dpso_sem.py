from __future__ import division

from sre_parse import Tokenizer

import numpy as np
import tensorflow as tf

import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras_preprocessing.text import Tokenizer
import data_helper
import glove_utils
from models.QNN import QNN
from models.TextCNN import TextCNN
from utils import vector_to_text

np.random.seed(1300)
tf.set_random_seed(1300)
VOCAB_SIZE = 50000
with open('PSO/word_candidates_sense.pkl', 'rb') as fp:
    word_candidate = pickle.load(fp)
with open('PSO/pos_tags_test.pkl', 'rb') as fp:
    test_pos_tags = pickle.load(fp)

# Prevent returning 0 as most similar word because it is not part of the dictionary
max_len = 200
batch_size = 1
lstm_size = 128
# max_len =  100

from attack_dpso_sem import PSOAttack

pop_size = 60
train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_texts)
glove_model = glove_utils.loadGloveModel('glove.6B.50d.txt')
embedding_matrix, _ = glove_utils.create_embeddings_matrix(glove_model, tokenizer.word_index, 50,
                                                           VOCAB_SIZE)
embedding_matrix = np.transpose(embedding_matrix)
model = QNN(embedding_matrix, tokenizer, max_len, 50, VOCAB_SIZE, 2)
model.load('checkpoint/qnn_our.hdf5')
# model = TextCNN(embedding_matrix, tokenizer, max_len, 50, VOCAB_SIZE)
# model.load('checkpoint/cnn.hdf5')

f = open('PSO/fool_result/adv_%s.txt' % model.__class__.__name__, mode='w+', encoding='utf-8')
f_clean = open('PSO/fool_result/clean.txt', mode='w+', encoding='utf-8')
ga_atttack = PSOAttack(model, word_candidate, tokenizer,
                       max_iters=20,
                       pop_size=pop_size)
SAMPLE_SIZE = len(test_labels)
TEST_SIZE = 1000
test_x = tokenizer.texts_to_sequences(test_texts)
test_x = pad_sequences(test_x, maxlen=max_len, dtype='int32', padding='post', truncating='post')
test_y = np.array(test_labels)
test_idx = np.random.choice(len(test_labels), SAMPLE_SIZE, replace=False)

test_list = []
orig_list = []
orig_label_list = []
adv_list = []
dist_list = []
adv_orig = []
adv_orig_label = []
fail_list = []
adv_training_examples = []
SUCCESS_THRESHOLD = 0.25
for i in range(SAMPLE_SIZE):
    pos_tags = test_pos_tags[test_idx[i]]
    x_orig = test_x[test_idx[i]]
    orig_label = test_y[test_idx[i]]
    orig_preds = model.predict(vector_to_text(x_orig[np.newaxis, :], tokenizer))[0]
    # orig_preds = model.predict(x_orig[np.newaxis, :])[0]
    if np.argmax(orig_preds) != orig_label:
        print('skipping wrong classifed ..')
        print('--------------------------')
        continue
    x_len = np.sum(np.sign(x_orig))
    # if x_len >= 100:
    #     print('skipping too long input..')
    #     print('--------------------------')
    #     continue
    # if x_len < 10:
    #     print('skipping too short input..')
    #     print('--------------------------')
    #     continue
    print('****** ', len(test_list) + 1, ' ********')
    test_list.append(test_idx[i])
    orig_list.append(x_orig)
    target_label = 1 if orig_label == 0 else 0
    orig_label_list.append(orig_label)
    x_adv = ga_atttack.attack(x_orig, target_label, pos_tags)
    f_clean.write(test_texts[test_idx[i]] + '\t' + str(orig_label) + '\n')
    if x_adv is None:
        print('%d failed' % (i + 1))
        f.write(vector_to_text([x_orig], tokenizer)[0] + '\t' + str(orig_label) + '\n')
    else:
        num_changes = np.sum(x_orig != x_adv)
        print('%d - %d changed.' % (i + 1, int(num_changes)))
        modify_ratio = num_changes / x_len
        if modify_ratio > 0.25:
            print('too long:', modify_ratio)
            f.write(vector_to_text([x_orig], tokenizer)[0] + '\t' + str(orig_label) + '\n')
        else:
            print('success!')
            f.write(vector_to_text([x_adv], tokenizer)[0] + '\t' + str(orig_label) + '\n')

        # display_utils.visualize_attack(sess, model, dataset, x_orig, x_adv)
    print('--------------------------')
    if (len(test_list) >= TEST_SIZE):
        break
f.close()
