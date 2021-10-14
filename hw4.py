from pymystem3 import Mystem
from string import punctuation
import nltk
from nltk.corpus import stopwords
import re
from scipy import sparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import pickle
from gensim.models import KeyedVectors
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

m = Mystem()
nltk.download('stopwords')
stopword = stopwords.words('russian')
punctuation += '...' + '—' + '…' + '«»'


def first_processing(file):
    with open(file, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]

    answers_corpus = []
    dropped = []

    for i, part in enumerate(corpus):
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
            dropped.append(i)
            pass

    questions_corpus = []

    for i, part in enumerate(corpus):
        if i not in dropped:
            questions_corpus.append(json.loads(part)['question'])

    return answers_corpus, questions_corpus


def second_processing(given_corpus):
    corpus = []
    dropped = []
    for text in given_corpus:
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
        if text == '':
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


def get_cleared_corpus(answers_corpus, questions_corpus):
    cleared_answers_corpus, ans_dropped = second_processing(answers_corpus)

    for ind in ans_dropped:
        del questions_corpus[ind]

    for ind in ans_dropped:
        del answers_corpus[ind]

    cleared_questions_corpus, ques_dropped = second_processing(questions_corpus)

    for ind in ques_dropped:
        del cleared_answers_corpus[ind]

    for ind in ques_dropped:
        del questions_corpus[ind]

    for ind in ques_dropped:
        del answers_corpus[ind]

    return answers_corpus, questions_corpus, cleared_answers_corpus, cleared_questions_corpus


def fasttext_get_matrix(texts):
    fasttext_vectors = []
    for i, text in enumerate(texts):
        tokens = text.split()
        tokens_vectors = np.zeros((len(tokens), fasttext_model.vector_size))
        for i, token in enumerate(tokens):
            tokens_vectors[i] = fasttext_model[token]
        if tokens_vectors.shape[0] != 0:
            means = np.mean(tokens_vectors, axis=0)
            n_means = means / np.linalg.norm(means)

        fasttext_vectors.append(n_means)

    return sparse.csr_matrix(fasttext_vectors)


def get_similarity(sparced_matrix, query_vec):
    scores = cosine_similarity(sparced_matrix, query_vec)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return list(np.array(answers_corpus)[sorted_scores_indx.ravel()][:5])


def fasttext_query_preprocessing(query):
    query = [word.lower().strip(punctuation).strip() for word in query.split()]
    query = m.lemmatize(' '.join([word for word in query]))
    query = ' '.join([word for word in query])
    tokens = [word for word in query.split() if word != '']
    query_vectors = []
    tokens_vectors = np.zeros((len(tokens), fasttext_model.vector_size))

    for i, token in enumerate(tokens):
        tokens_vectors[i] = fasttext_model[token]
    if tokens_vectors.shape[0] != 0:
        means = np.mean(tokens_vectors, axis=0)
    n_means = means / np.linalg.norm(means)
    query_vectors.append(n_means)

    return sparse.csr_matrix(query_vectors)


def fasttext_search():
    while True:
        query = input('Введите запрос (или "ОСТАНОВИТЕ" для остановки):')
        if query == 'ОСТАНОВИТЕ':
            break
        query_vec = fasttext_query_preprocessing(query)
        pprint(get_similarity(fasttext_ques_matrix, query_vec))


# принимает спарс матрицы
def scoring(q_matrix, a_matrix):
    q_matrix = np.delete(q_matrix.toarray(), np.s_[10000:], 0)
    a_matrix = np.delete(a_matrix.toarray(), np.s_[10000:], 0)

    scoring_matrix = np.dot(q_matrix, a_matrix.T)
    score = 0
    for ind, line in enumerate(scoring_matrix):

        sorted_scores_indx = np.argsort(line, axis=0)[::-1]
        sorted_scores_indx = [sorted_scores_indx.ravel()][0][:5]
        if ind in sorted_scores_indx:
            score += 1

    return score / 10000


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def bert_vectorizer(corpus):
    corpus = [corpus[i:i + 3250] for i in range(0, len(corpus), 3250)]
    bert_vects = []
    for text in tqdm(corpus):
        encoded_input = auto_tokenizer(text, padding=True, truncation=True, max_length=24, return_tensors='pt')
        encoded_input = encoded_input.to('cuda')
        with torch.no_grad():
            model_output = auto_model(**encoded_input)

        sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings)
        for sentence in sentence_embeddings:
            bert_vects.append(sentence.cpu().numpy())
    torch.cuda.empty_cache()

    return sparse.csr_matrix(bert_vects)


def bert_search():
    while True:
        query = input('Введите запрос (или "ОСТАНОВИТЕ" для остановки):')
        if query == 'ОСТАНОВИТЕ':
            break
        query_vec = bert_vectorizer(query)
        pprint(get_similarity(b_questions, query_vec))


def bm25_query_preprocessing(query, count_vectorizer):
    query = [word.lower().strip(punctuation).strip() for word in query.split()]
    query = m.lemmatize(' '.join([word for word in query]))
    query = ' '.join([word for word in query])
    query = ' '.join([word for word in query.split() if word != ''])
    query_vec = count_vectorizer.transform([query])
    return query_vec


def bm25_vectorization(ans_cleared_corpus, que_cleared_corpus):
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(ans_cleared_corpus)  # для индексации запроса
    x_tf_vec = tf_vectorizer.fit_transform(ans_cleared_corpus)  # матрица с tf
    x_tfidf_vec = tfidf_vectorizer.fit_transform(ans_cleared_corpus)  # матрица для idf
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

    bm25_answers = sparse.csr_matrix((values, (rows, cols)))
    bm25_questions = count_vectorizer.transform(que_cleared_corpus)
    return bm25_answers, bm25_questions, count_vectorizer


def bm25_tfidf_count_search(vectorizer):
    while True:
        query = input('Введите запрос (или "ОСТАНОВИТЕ" для остановки):')
        if query == 'ОСТАНОВИТЕ':
            break
        query_vec = bm25_query_preprocessing(query, vectorizer)
        pprint(get_similarity(bm25_questions, query_vec))


# я тут сразу на поиск даю 10к, потому что не хватает оперативы
def scoring_for_count(q_matrix, a_matrix):
    # q_matrix = np.delete(q_matrix., np.s_[5000:], 0)
    # a_matrix = np.delete(a_matrix.toarray(), np.s_[5000:], 0)
    scoring_matrix = np.dot(q_matrix.toarray(), a_matrix.T.toarray())
    score = 0
    for ind, line in enumerate(scoring_matrix):
        sorted_scores_indx = np.argsort(line, axis=0)[::-1]
        sorted_scores_indx = [sorted_scores_indx.ravel()][0][:5]
        if ind in sorted_scores_indx:
            score += 1

    return score / 10000


def tfidf_vectorization(ans_cleared_corpus, que_cleared_corpus):
    vectorizer = TfidfVectorizer()
    tfidf_answers = vectorizer.fit_transform(ans_cleared_corpus)
    tfidf_questions = vectorizer.transform(que_cleared_corpus)

    return tfidf_answers, tfidf_questions, vectorizer


def count_vectorization(ans_cleared_corpus, que_cleared_corpus):
    vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    count_answers = vectorizer.fit_transform(ans_cleared_corpus)
    count_questions = vectorizer.transform(que_cleared_corpus)

    return count_answers, count_questions, vectorizer


if __name__ == '__main__':
    # для создания корпусов
    # answers_corpus, questions_corpus = first_processing('questions_about_love.jsonl')
    # answers_corpus, questions_corpus, cleared_answers_corpus, cleared_questions_corpus = get_cleared_corpus(answers_corpus, questions_corpus
    # with open('answers_corpus.pickle', 'wb') as f:
    #     pickle.dump(answers_corpus, f)
    # with open('questions_corpus.pickle', 'wb') as f:
    #     pickle.dump(questions_corpus, f)
    # with open('cleared_answers_corpus.pickle', 'wb') as f:
    #     pickle.dump(cleared_answers_corpus, f)
    # with open('cleared_questions_corpus.pickle', 'wb') as f:
    #     pickle.dump(cleared_questions_corpus, f)

    with open('answers_corpus.pickle', 'rb') as f:
        answers_corpus = pickle.load(f)
    with open('questions_corpus.pickle', 'rb') as f:
        questions_corpus = pickle.load(f)
    with open('cleared_answers_corpus.pickle', 'rb') as f:
        cleared_answers_corpus = pickle.load(f)
    with open('cleared_questions_corpus.pickle', 'rb') as f:
        cleared_questions_corpus = pickle.load(f)

    # fasttext module
    # так же закоменченно все что уже готово
    fasttext_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    # fasttext_ans_matrix = fasttext_get_matrix(cleared_answers_corpus)
    # fasttext_ques_matrix = fasttext_get_matrix(cleared_questions_corpus)
    # sparse.save_npz('fasttext_ans_matrix.npz', fasttext_ans_matrix)
    fasttext_ans_matrix = sparse.load_npz('fasttext_ans_matrix.npz')
    # sparse.save_npz('fasttext_ques_matrix.npz', fasttext_ques_matrix)
    fasttext_ques_matrix = sparse.load_npz('fasttext_ques_matrix.npz')
    print('Fasttext result: ', scoring(fasttext_ques_matrix, fasttext_ans_matrix))

    auto_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_model.to('cuda')
    # torch.save(bert_vectorizer(questions_corpus), 'ques_bert.pt')
    # torch.save(bert_vectorizer(answers_corpus), 'ans_bert.pt')
    b_answers = torch.load('ans_bert.pt')
    b_questions = torch.load('ques_bert.pt')
    print('Bert result: ', scoring(b_questions, b_answers))

    # BM25
    bm25_answers, bm25_questions, bm25_count_vectorizer = bm25_vectorization(cleared_answers_corpus[:10000],
                                                                             cleared_questions_corpus[:10000])
    print('BM25 result: ', scoring_for_count(bm25_questions, bm25_answers))

    # TF IDF
    tfidf_answers, tfidf_questions, tfidf_vectorizer = tfidf_vectorization(cleared_answers_corpus[:10000],
                                                                           cleared_questions_corpus[:10000])

    print('TF IDF result: ', scoring_for_count(tfidf_questions, tfidf_answers))

    # Count Vectorizer
    count_answers, count_questions, count_vectorizer = count_vectorization(cleared_answers_corpus[:10000],
                                                                           cleared_questions_corpus[:10000])
    print('Count Vectorizer result: ', scoring_for_count(count_questions, count_answers))

    while True:
        jaustalblin = input('Введите что-нибудь: fasttext, bert, bm25, tfidf, count')
        if jaustalblin == 'fasttext':
            fasttext_search()
        elif jaustalblin == 'bert':
            bert_search()
        elif jaustalblin == 'bm25':
            bm25_tfidf_count_search(bm25_count_vectorizer)
        elif jaustalblin == 'tfidf':
            bm25_tfidf_count_search(tfidf_vectorizer)
        elif jaustalblin == 'count':
            bm25_tfidf_count_search(count_vectorizer)
        else:
            pass
