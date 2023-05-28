# -*- coding: utf-8 -*-
import spacy
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

from PWWS.config import config

nlp = spacy.load('en_core_web_sm')


def text_to_vector(text, tokenizer, dataset):
    vector = tokenizer.texts_to_sequences([text])
    vector = pad_sequences(vector, maxlen=config.word_max_len[dataset], padding='post', truncating='post')
    return vector


def text_to_vector_for_all(text_list, tokenizer, dataset):
    vector = tokenizer.texts_to_sequences(text_list)
    vector = pad_sequences(vector, maxlen=config.word_max_len[dataset], padding='post', truncating='post')
    return vector


def vector_to_text(vector, tokenizer):
    word_index = tokenizer.word_index
    reverse_word_index = dict()
    for word in word_index.keys():
        index = word_index.get(word)
        reverse_word_index[index] = word
    reverse_word_index[0] = 'UNK'
    arr = [reverse_word_index.get(i) for i in vector[0]]
    result = [' '.join(arr)]
    return result
