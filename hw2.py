from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from pymystem3 import Mystem
from string import punctuation
punctuation += '...' + '—' + '…' + '«»'
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
from pprint import pprint
from tqdm import tqdm

#
def get_paths():
    curr_dir = os.getcwd()
    sub_dir = os.path.join(curr_dir, 'friends-data')

    paths = []
    names = []
    for root, dirs, files in os.walk(curr_dir):
        for name in files:
            if name.endswith('.ru.txt'):
                paths.append(os.path.join(root, name))
                names.append(name)

    return paths, names


# препроцессинг данных (медленный, но качественный)
def preproc(paths):
    mystem = Mystem()
    corpus = []
    for path in tqdm(paths):
        with open(path, 'r', encoding='utf-8-sig') as f:
            text = f.read()
        text = re.sub('[0-9a-zA-Z]+', '', text)
        text = [word.lower().strip().strip(punctuation) for word in text.split()]
        text = mystem.lemmatize(' '.join([x for x in text]))
        text = [word for word in text if word != '']
        text = [word for word in text if word != r' ']
        text = [word for word in text if word != r'  ']
        text = ' '.join([x for x in text if x not in stopwords.words('russian')])
        corpus.append(text)

    return corpus


# функция индексации корпуса, на выходе которой посчитанная матрица Document-Term
def vect_corpus(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    return vectorizer, X


# обработка запроса
def trans_query(query, vectorizer):
    mystem = Mystem()
    query = re.sub('[0-9a-zA-Z]+', '', query)
    query = [word.lower().strip().strip(punctuation) for word in query.split()]
    query = mystem.lemmatize(' '.join([x for x in query]))
    query = [word for word in query if word != '']
    query = ' '.join([x for x in query if x not in stopwords.words('russian')])

    return vectorizer.transform([query]).toarray()


# перемножение (нахождение косинусной близости)
def cos_counter(query, vectorizer, v_corpus):
    query = trans_query(query, vectorizer)
    cos_sim = v_corpus.toarray().dot(query.transpose())

    return cos_sim


# поиск релевантных док-ов
def doc_finder(cos_result):
    sorted_сos_result = np.argsort(cos_result, axis=0)
    reversed_scr = np.flip(sorted_сos_result)
    result = []
    for ind in reversed_scr:
        result.append(names[int(ind)])

    return result


if __name__ == '__main__':
    paths, names = get_paths()
    corpus = preproc(paths)
    vectorizer, v_corpus = vect_corpus(corpus)

    while True:
        query = input('Введите запрос:')
        cos_result = cos_counter(query, vectorizer, v_corpus)
        result = doc_finder(cos_result)
        pprint(result)