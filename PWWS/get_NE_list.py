# coding: utf-8
import argparse
import copy
from collections import defaultdict

import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')
parser = argparse.ArgumentParser('named entity recognition')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='yahoo')

NE_type_dict = {
    'PERSON': defaultdict(int),  # People, including fictional.
    'NORP': defaultdict(int),  # Nationalities or religious or political groups.
    'FAC': defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    'ORG': defaultdict(int),  # Companies, agencies, institutions, etc.
    'GPE': defaultdict(int),  # Countries, cities, states.
    'LOC': defaultdict(int),  # Non-GPE locations, mountain ranges, bodies of water.
    'PRODUCT': defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    'EVENT': defaultdict(int),  # Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART': defaultdict(int),  # Titles of books, songs, etc.
    'LAW': defaultdict(int),  # Named documents made into laws.
    'LANGUAGE': defaultdict(int),  # Any named language.
    'DATE': defaultdict(int),  # Absolute or relative dates or periods.
    'TIME': defaultdict(int),  # Times smaller than a day.
    'PERCENT': defaultdict(int),  # Percentage, including "%".
    'MONEY': defaultdict(int),  # Monetary values, including unit.
    'QUANTITY': defaultdict(int),  # Measurements, as of weight or distance.
    'ORDINAL': defaultdict(int),  # "first", "second", etc.
    'CARDINAL': defaultdict(int),  # Numerals that do not fall under another type.
}


def recognize_named_entity(texts):
    '''
    Returns all NEs in the input texts and their corresponding types
    '''
    NE_freq_dict = copy.deepcopy(NE_type_dict)

    for text in texts:
        doc = nlp(text)
        for word in doc.ents:
            NE_freq_dict[word.label_][word.text] += 1
    return NE_freq_dict


class NameEntityList(object):
    # If the original input in IMDB belongs to class 0 (negative)
    imdb_0 = {'PERSON': 'David',
              'NORP': 'Australian',
              'FAC': 'Hound',
              'ORG': 'Ford',
              'GPE': 'India',
              'LOC': 'Atlantic',
              'PRODUCT': 'Highly',
              'EVENT': 'Depression',
              'WORK_OF_ART': 'Casablanca',
              'LAW': 'Constitution',
              'LANGUAGE': 'Portuguese',
              'DATE': '2001',
              'TIME': 'hours',
              'PERCENT': '98%',
              'MONEY': '4',
              'QUANTITY': '70mm',
              'ORDINAL': '5th',
              'CARDINAL': '7',
              }
    # If the original input in IMDB belongs to class 1 (positive)
    imdb_1 = {'PERSON': 'Lee',
              'NORP': 'Christian',
              'FAC': 'Shannon',
              'ORG': 'BAD',
              'GPE': 'Seagal',
              'LOC': 'Malta',
              'PRODUCT': 'Cat',
              'EVENT': 'Hugo',
              'WORK_OF_ART': 'Jaws',
              'LAW': 'RICO',
              'LANGUAGE': 'Sebastian',
              'DATE': 'Friday',
              'TIME': 'minutes',
              'PERCENT': '75%',
              'MONEY': '$',
              'QUANTITY': '9mm',
              'ORDINAL': 'sixth',
              'CARDINAL': 'zero',
              }
    imdb = [imdb_0, imdb_1]
    agnews_0 = {'PERSON': 'Williams',
                'NORP': 'European',
                'FAC': 'Olympic',
                'ORG': 'Microsoft',
                'GPE': 'Australia',
                'LOC': 'Earth',
                'PRODUCT': '#',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Chinese',
                'DATE': 'third-quarter',
                'TIME': 'Tonight',
                'MONEY': '#39;t',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '1',
                }
    agnews_1 = {'PERSON': 'Bush',
                'NORP': 'Iraqi',
                'FAC': 'Outlook',
                'ORG': 'Microsoft',
                'GPE': 'Iraq',
                'LOC': 'Asia',
                'PRODUCT': '#',
                'EVENT': 'Series',
                'WORK_OF_ART': 'Nobel',
                'LAW': 'Constitution',
                'LANGUAGE': 'French',
                'DATE': 'third-quarter',
                'TIME': 'hours',
                'MONEY': '39;Keefe',
                'ORDINAL': '2nd',
                'CARDINAL': 'Two',
                }
    agnews_2 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Baghdad',
                'LOC': 'Earth',
                'PRODUCT': 'Soyuz',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Constitution',
                'LANGUAGE': 'Filipino',
                'DATE': 'Sunday',
                'TIME': 'evening',
                'MONEY': '39;m',
                'QUANTITY': '20km',
                'ORDINAL': 'eighth',
                'CARDINAL': '6',
                }
    agnews_3 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Iraq',
                'LOC': 'Kashmir',
                'PRODUCT': 'Yukos',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'Gazprom',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Hebrew',
                'DATE': 'Saturday',
                'TIME': 'overnight',
                'MONEY': '39;m',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '6',
                }
    agnews = [agnews_0, agnews_1, agnews_2, agnews_3]
    sst_0 = {'PERSON': 'David',
             'NORP': 'Australian',
             'FAC': 'Hound',
             'ORG': 'Ford',
             'GPE': 'India',
             'LOC': 'Atlantic',
             'PRODUCT': 'Highly',
             'EVENT': 'Depression',
             'WORK_OF_ART': 'Casablanca',
             'LAW': 'Constitution',
             'LANGUAGE': 'Portuguese',
             'DATE': '2001',
             'TIME': 'hours',
             'PERCENT': '98%',
             'MONEY': '4',
             'QUANTITY': '70mm',
             'ORDINAL': '5th',
             'CARDINAL': '7',
             }

    sst_1 = {'PERSON': 'Lee',
             'NORP': 'Christian',
             'FAC': 'Shannon',
             'ORG': 'BAD',
             'GPE': 'Seagal',
             'LOC': 'Malta',
             'PRODUCT': 'Cat',
             'EVENT': 'Hugo',
             'WORK_OF_ART': 'Jaws',
             'LAW': 'RICO',
             'LANGUAGE': 'Sebastian',
             'DATE': 'Friday',
             'TIME': 'minutes',
             'PERCENT': '75%',
             'MONEY': '$',
             'QUANTITY': '9mm',
             'ORDINAL': 'sixth',
             'CARDINAL': 'zero',
             }
    sst = [sst_0, sst_1]
    L = {'imdb': imdb, 'agnews': agnews, 'sst': sst}


NE_list = NameEntityList()
