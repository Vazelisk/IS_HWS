from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import nltk
import re
from scipy import sparse
import numpy as np
import json

punctuation += '...' + '—' + '…' + '«»'
nltk.download('stopwords')
stopword = stopwords.words('russian')
m = Mystem()


def first_processing(file):
    with open(file, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]

    answers_corpus = []

    for part in corpus:
        d = dict()
        j_answers = json.loads(part)['answers']
        try:  # бывает пустое поле answers
            for ind, gr_comments in enumerate(j_answers):

                try:
                    d[ind] = int(gr_comments['author_rating']['value'])

                except ValueError:  # бывает пустое поле value
                    d[ind] = 0

            ind = sorted(d.items(), key=lambda item: item[1], reverse=True)[0][0]
            answers_corpus.append(j_answers[ind]['text'])

        except IndexError:
            pass

    return answers_corpus


def second_processing(cleared_corpus):
    corpus = []
    dropped = []
    for text in cleared_corpus:
        text = re.sub('[0-9a-zA-Z]+', '', text)
        text = [word.lower().strip().strip(punctuation) for word in text.split()]
        # text = [x for x in text if x not in stopword]
        text = ' '.join([word for word in text if word != ''])
        corpus.append(text)

    # чтобы удалить пустые строки и сохранить их индексы
    # чтобы убрать их в изначальном датасете
    for ind, text in enumerate(corpus):
        if not text:
            dropped.append(ind)
            del corpus[ind]

    lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
    txtpart = lol(corpus, 1000)
    res = []
    for txtp in txtpart:
        alltexts = ' '.join([txt + ' br ' for txt in txtp])
        words = m.lemmatize(alltexts)
        doc = []
        for txt in words:
            if txt != '\n' and txt.strip() != '':
                if txt == 'br':
                    res.append(' '.join(doc))
                    doc = []
                else:
                    doc.append(txt)

    return res, dropped


def vectorization(cleared_corpus):
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(cleared_corpus)  # для индексации запроса
    x_tf_vec = tf_vectorizer.fit_transform(cleared_corpus)  # матрица с tf
    x_tfidf_vec = tfidf_vectorizer.fit_transform(cleared_corpus)  # матрица для idf

    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vec

    values = []
    rows = []
    cols = []
    k = 2
    b = 0.75

    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d / avdl))

    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        A = idf[0][j] * tf[i, j] * k + 1
        B = tf[i, j] + B_1[i]
        AB = A / B

        values.append(np.asarray(AB)[0][0])

    return sparse.csr_matrix((values, (rows, cols))), count_vectorizer


def query_preprocessing(query, count_vectorizer):
    query = [word.lower().strip(punctuation).strip() for word in query.split()]
    query = m.lemmatize(' '.join([word for word in query]))
    query = ' '.join([word for word in query])
    query = ' '.join([word for word in query.split() if word != ''])
    query_vec = count_vectorizer.transform([query])
    return query_vec


def search(sparced_matrix, query_vec):
    scores = np.dot(sparced_matrix, query_vec.T)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return list(np.array(answers_corpus)[sorted_scores_indx.ravel()][:10])


if __name__ == '__main__':
    answers_corpus = first_processing('questions_about_love.jsonl')
    cleared_corpus, dropped = second_processing(answers_corpus)
    # удаляю удаленные при лемматизации строки
    for ind in dropped:
        del answers_corpus[ind]

    sparced_matrix, count_vectorizer = vectorization(cleared_corpus)

    while True:
        query = input('Введите запрос (или "ОСТАНОВИТЕ" для остановки):')
        if query == 'ОСТАНОВИТЕ':
            break
        query_vec = query_preprocessing(query, count_vectorizer)
        pprint(search(sparced_matrix, query_vec))
