# coding: utf-8
from PWWS.config import config
import copy
import spacy
from PWWS.word_level_process import text_to_vector, vector_to_text

nlp = spacy.load('en_core_web_sm')


def evaluate_word_saliency(doc, grad_guide, tokenizer, input_y, dataset, level):
    word_saliency_list = []

    # zero the code of the current word and calculate the amount of change in the classification probability
    if level == 'word':
        max_len = config.word_max_len[dataset]
        text = [doc[position].text for position in range(len(doc))]
        text = ' '.join(text)
        origin_vector = text_to_vector(text, tokenizer, dataset)
        origin_prob = grad_guide.predict_prob([text])
        for position in range(len(doc)):
            if position >= max_len:
                break
            # get x_i^(\hat)
            without_word_vector = copy.deepcopy(origin_vector)
            without_word_vector[0][position] = 0

            prob_without_word = grad_guide.predict_prob(vector_to_text(without_word_vector, tokenizer))

            # calculate S(x,w_i) defined in Eq.(6)
            word_saliency = origin_prob[input_y] - prob_without_word[input_y]
            word_saliency_list.append((position, doc[position], word_saliency, doc[position].tag_))
    position_word_list = []
    for word in word_saliency_list:
        position_word_list.append((word[0], word[1]))

    return position_word_list, word_saliency_list
