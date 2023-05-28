import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import pickle


def split(text, num):
    length = len(text)
    r = [[] for i in range(num)]
    for i in range(num):
        r[i] = text[i * int(length / num):(i + 1) * int(length / num)]
    return np.array(r)


def split_texts(texts, num, seq_length, tokenizer):
    results = [[] for i in range(num)]
    for i in range(len(texts)):
        sentences = split(texts[i], num)
        for j in range(num):
            if j >= len(sentences):
                results[j].append(' ')
                continue
            results[j].append(sentences[j])

    for i in range(len(results)):
        tmp = tokenizer.texts_to_sequences(results[i])
        tmp = pad_sequences(tmp, maxlen=seq_length, dtype='int32', padding='post', truncating='post')
        results[i] = tmp
    return results


def dump(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
