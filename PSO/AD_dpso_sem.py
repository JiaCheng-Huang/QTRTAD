from __future__ import division

import os
import pickle

import tensorflow as tf
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

import data_helper
import glove_utils
from models import *
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
TEST_SIZE = 1000
train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_texts)
glove_model = glove_utils.loadGloveModel('glove.6B.50d.txt')
embedding_matrix, _ = glove_utils.create_embeddings_matrix(glove_model, tokenizer.word_index, 50,
                                                           VOCAB_SIZE)
embedding_matrix = np.transpose(embedding_matrix)
# model = QNN(embedding_matrix, tokenizer, max_len, 50, VOCAB_SIZE, 5)
# model = TextCNN(embedding_matrix, tokenizer, max_len, 50, VOCAB_SIZE)
# model = BidLSTM(embedding_matrix, tokenizer, max_len, 50, VOCAB_SIZE)
model = PlainQNN(embedding_matrix, tokenizer, max_len, 50, VOCAB_SIZE)
# model = Transformer(embedding_matrix, tokenizer, max_len, 50, VOCAB_SIZE)
model_name = model.__class__.__name__
model.load('checkpoint/%s.hdf5' % model_name)

adv_text_path = 'PSO/fool_result/adv_%s.csv' % model_name
if not os.path.exists(adv_text_path):
    file = open(adv_text_path, "a+", encoding='utf-8')
    file.write('clean' + '\t' + 'adv' + '\t' + 'label' + "\n")
    test_texts = test_texts[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]
    test_pos_tags = test_pos_tags[:TEST_SIZE]
else:
    file = open(adv_text_path, "r+", encoding='utf-8')
    lines = len(file.readlines())
    test_texts = test_texts[lines - 1:TEST_SIZE]
    test_labels = test_labels[lines - 1:TEST_SIZE]
    test_pos_tags = test_pos_tags[lines - 1:TEST_SIZE]

ga_atttack = PSOAttack(model, word_candidate, tokenizer,
                       max_iters=20,
                       pop_size=pop_size)
SAMPLE_SIZE = len(test_labels)

test_x = tokenizer.texts_to_sequences(test_texts)
test_x = pad_sequences(test_x, maxlen=max_len, dtype='int32', padding='post', truncating='post')
test_y = np.array(test_labels)
# test_idx = np.random.choice(len(test_labels), SAMPLE_SIZE, replace=False)

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
    pos_tags = test_pos_tags[i]
    x_orig = test_x[i]
    orig_label = test_y[i]
    # print(x_orig[np.newaxis,:])
    # print(vector_to_text(x_orig[np.newaxis, :], tokenizer)[0])
    # print(len(vector_to_text(x_orig[np.newaxis, :], tokenizer)[0].split(' ')))
    print('****** ', len(test_list) + 1, ' ********')
    x_len = np.sum(np.sign(x_orig))
    test_list.append(i)
    orig_list.append(x_orig)
    target_label = 1 if orig_label == 0 else 0
    orig_label_list.append(orig_label)
    orig_preds = model.predict(vector_to_text(x_orig[np.newaxis, :], tokenizer))[0]
    if np.argmax(orig_preds) != orig_label:
        adv_doc = vector_to_text([x_orig], tokenizer)[0]
        print('%d wrong classified.' % (i + 1))
    else:
        x_adv = ga_atttack.attack(x_orig, target_label, pos_tags)
        if x_adv is None:
            print('%d failed' % (i + 1))
            adv_doc = vector_to_text([x_orig], tokenizer)[0]
        else:
            num_changes = np.sum(x_orig != x_adv)
            print('%d - %d changed.' % (i + 1, int(num_changes)))
            modify_ratio = num_changes / x_len
            if modify_ratio > 0.25:
                print('too long:', modify_ratio)
                adv_doc = vector_to_text([x_orig], tokenizer)[0]
            else:
                print('success!')
                adv_doc = vector_to_text([x_adv], tokenizer)[0]

    file.write(vector_to_text([x_orig], tokenizer)[0] + '\t' + adv_doc + '\t' + str(orig_label) + "\n")
    print('--------------------------')
    if len(test_list) >= TEST_SIZE:
        break
file.close()
