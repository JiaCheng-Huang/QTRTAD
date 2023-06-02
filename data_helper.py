import csv
import os
import re
import numpy as np
np.random.seed(1205)

def read_imdb_files(filetype):
    """
    filetype: 'train' or 'test'
    """

    # [0,1] means positive，[1,0] means negative
    all_labels = []
    for _ in range(12500):
        all_labels.append(1)
    for _ in range(12500):
        all_labels.append(0)

    all_texts = []
    file_list = []
    path = r'data_set/aclImdb/'
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            txt = rm_tags(" ".join(f.readlines()))
            txt = txt.replace('  ', ' ')
            all_texts.append(txt)
    return all_texts, all_labels


def split_imdb_files():
    print('Processing IMDB dataset')
    train_texts, train_labels = read_imdb_files('train')
    test_texts, test_labels = read_imdb_files('test')
    return train_texts, train_labels, test_texts, test_labels


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_agnews_files(filetype):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    path = r'../data_set/ag_news_csv/{}.csv'.format(filetype)
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        content = line[1] + ". " + line[2]
        texts.append(content.replace('  ', ' '))
        labels_index.append(line[0])
        doc_count += 1

    # Start document processing
    labels = []
    for i in range(doc_count):
        label_class = np.zeros(4, dtype='float32')
        label_class[int(labels_index[i]) - 1] = 1
        labels.append(label_class)

    return texts, labels, labels_index


def split_agnews_files():
    print("Processing AG's News dataset")
    train_texts, train_labels, _ = read_agnews_files('train')  # 120000
    test_texts, test_labels, _ = read_agnews_files('test')  # 7600
    return train_texts, train_labels, test_texts, test_labels


def read_sst_2_files(filetype):
    texts = []
    labels = []
    path = r'../data_set/sst/{}.tsv'.format(filetype)
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter='	', quotechar='"'):
        texts.append(line[0])
        if line[1] == '1':
            labels.append([1, 0])
        elif line[1] == '0':
            labels.append([0, 1])
    return texts, labels


def split_sst_2_files():
    print("Processing SST-2 dataset")
    train_texts, train_labels = read_sst_2_files('train')
    test_texts, test_labels = read_sst_2_files('test')
    return train_texts, train_labels, test_texts, test_labels
