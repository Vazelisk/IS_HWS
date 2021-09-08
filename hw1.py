import os
import pymorphy2
from string import punctuation
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def get_paths():
    curr_dir = os.getcwd()
    sub_dir = os.path.join(curr_dir, 'friends-data')

    paths = []
    for root, dirs, files in os.walk(curr_dir):
        for name in files:
            if name.endswith('txt'):
                paths.append(os.path.join(root, name))

    return paths


def preproc(path):
    morph = pymorphy2.MorphAnalyzer()
    with open(path, 'r', encoding='utf-8-sig') as f:
        text = f.read()
        text = [word.lower().strip(punctuation) for word in text.split()]
        text = [word for word in text if word not in stopwords.words('russian')]
        text = [word for word in text if word != '']
        del text[-4:]  # meta info

    lemmas = str()
    known_words = {}

    for word in text:
        if word in known_words:
            lemmas += (known_words[word] + ' ')
        else:
            result = morph.parse(word)[0].normal_form
            lemmas += (result + ' ')
            known_words[word] = result

    return lemmas


def vectorizing(paths):
    texts = []
    for path in paths:
        processed = preproc(path)
        texts.append(processed)

    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(texts).todense()
    data = pd.DataFrame(X, columns=vectorizer.get_feature_names())

    return data


def get_mms(data):
    summarizer = data.sum()
    max_value = summarizer[summarizer == summarizer.max()]
    min_value = summarizer[summarizer == summarizer.min()]

    return ('Max:', max_value), ('Min:', min_value)


def get_inclusive(data):
    inclusive = list()
    for word, index in data.iteritems():
        temp = []
        for item in index:
            if item > 0:
                temp.append(word)
        if len(temp) == 165:
            inclusive.append(temp[0])

    return inclusive


def get_charac(data):
    charac = {
        'Моника': 0,
        'Рэйчел': 0,
        'Чендлер': 0,
        'Фиби': 0,
        'Росс': 0,
        'Джоуи': 0
    }

    for index, numbers in data.iteritems():
        if index == 'моника' or index == 'мон':
            for number in numbers:
                charac['Моника'] += number

        if index == 'рэйчел' or index == 'рейч':
            for number in numbers:
                charac['Рэйчел'] += number

        if index == 'чендлер' or index == 'чэндлер' or index == 'чен':
            for number in numbers:
                charac['Чендлер'] += number

        if index == 'фиби' or index == 'фибс':
            for number in numbers:
                charac['Фиби'] += number

        if index == 'росс':
            for number in numbers:
                charac['Росс'] += number

        if index == 'Джоуи' or index == 'джои' or index == 'джо':
            for number in numbers:
                charac['Джоуи'] += number

    return max(charac, key=charac.get), max(charac.values())


if __name__ == '__main__':
    paths = get_paths()
    data = vectorizing(paths)
    print('Самые частотное и редкие слова:', get_mms(data))
    print('Набор слов во всей коллекции:', get_inclusive(data))
    print('Самый часто встречающийся персонаж', get_charac(data))
