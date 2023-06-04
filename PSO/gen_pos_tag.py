import pickle

from nltk.tag import StanfordPOSTagger

import data_helper

train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
test_texts = test_texts[:1000]
train_text = [text.split(' ') for text in train_texts]
test_text = [text.split(' ') for text in test_texts]

jar = 'PSO/stanford-postagger-2018-10-16/stanford-postagger.jar'
model = 'PSO/stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

# from nltk.stem import WordNetLemmatizer
# wnl = WordNetLemmatizer()
# pos_tag=pos_tagger.tag(['what', "'s", 'invigorating', 'about', 'it', 'is', 'that', 'it', 'does', "n't", 'give', 'a', 'damn'])
# print(pos_tag)


all_pos_tags = []
test_pos_tags = []
i = 0
for text in train_text:
    i += 1
    print(i / len(train_text))
    pos_tags = pos_tagger.tag(text)
    all_pos_tags.append(pos_tags)
i = 0
for text in test_text:
    i += 1
    print(i / len(test_text))
    pos_tags = pos_tagger.tag(text)
    test_pos_tags.append(pos_tags)
f = open('PSO/pos_tags.pkl', 'wb')
pickle.dump(all_pos_tags, f)
f = open('PSO/pos_tags_test.pkl', 'wb')
pickle.dump(test_pos_tags, f)
