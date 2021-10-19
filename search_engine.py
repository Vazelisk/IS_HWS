from pymystem3 import Mystem
from string import punctuation
import nltk
from nltk.corpus import stopwords
from scipy import sparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import pickle
from gensim.models import KeyedVectors
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from time import time
from preparing_files import create_all
from get_scores import scoring

m = Mystem()
nltk.download('stopwords')
stopword = stopwords.words('russian')
punctuation += '...' + '—' + '…' + '«»'


def get_similarity(sparced_matrix, query_vec):
    scores = np.dot(sparced_matrix, query_vec.T)
    sorted_scores_indx = np.argsort(scores.toarray(), axis=0)[::-1]
    return list(np.array(answers_corpus)[sorted_scores_indx.ravel()][:5])


def fasttext_query_preprocessing(query):
    query = [word.lower().strip(punctuation).strip() for word in query.split()]
    query = m.lemmatize(' '.join([word for word in query]))
    query = ' '.join([word for word in query])
    tokens = [word for word in query.split() if word != '']
    return tokens


def fasttext_search():
    while True:
        query = input('Введите запрос (или "ОСТАНОВИТЕ" для остановки):')
        if query == 'ОСТАНОВИТЕ':
            break
        start = time()
        tokens = fasttext_query_preprocessing(query)
        query_vectors = []
        tokens_vectors = np.zeros((len(tokens), fasttext_model.vector_size))

        for i, token in enumerate(tokens):
            tokens_vectors[i] = fasttext_model[token]
        if tokens_vectors.shape[0] != 0:
            means = np.mean(tokens_vectors, axis=0)
            n_means = means / np.linalg.norm(means)
            query_vectors.append(n_means)
            end = time()
            print('Time spent for search - ', end - start)
            pprint(get_similarity(fasttext_ques_matrix, sparse.csr_matrix(query_vectors)))


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def bert_vectorizer(corpus):
    if corpus == '':
        return None
    else:
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
        start = time()
        query_vec = bert_vectorizer(query)
        if query_vec is None:
            continue
        else:
            pprint(get_similarity(b_questions, query_vec))
            end = time()
            print('Time spent for search - ', end - start)


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


def bm25_tfidf_count_search(vectorizer, questions_matrix):
    while True:
        query = input('Введите запрос (или "ОСТАНОВИТЕ" для остановки):')
        if query == 'ОСТАНОВИТЕ':
            break
        start = time()
        query_vec = bm25_query_preprocessing(query, vectorizer)
        pprint(get_similarity(questions_matrix, query_vec))
        end = time()
        print('Time spent for search - ', end - start)


if __name__ == '__main__':
    # uncomment to get matrixes
    create_all()
    start = time()
    print('loading corpuses...')
    with open('saved/answers_corpus.pickle', 'rb') as f:
        answers_corpus = pickle.load(f)

    # fasttext module
    print('loading fasttext...')
    fasttext_model = KeyedVectors.load('saved/araneum_none_fasttextcbow_300_5_2018.model')
    fasttext_ans_matrix = sparse.load_npz('saved/fasttext_ans_matrix.npz')
    fasttext_ques_matrix = sparse.load_npz('saved/fasttext_ques_matrix.npz')
    print('fasttext loaded!')

    print('loading bert...')
    auto_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_model.to('cuda')
    b_answers = torch.load('saved/ans_bert.pt')
    b_questions = torch.load('saved/ques_bert.pt')
    print('bert loaded!')

    # BM25
    print('loading bm25...')
    with open('saved/bm25_answers.npz', 'rb') as f:
        bm25_answers = pickle.load(f)
    with open('saved/bm25_questions.npz', 'rb') as f:
        bm25_questions = pickle.load(f)
    with open('saved/bm25_count_vectorizer.pickle', 'rb') as f:
        bm25_count_vectorizer = pickle.load(f)
    print('bm25 loaded!')

    # TF IDF
    print('loading TF IDF...')
    with open('saved/tfidf_answers.npz', 'rb') as f:
        tfidf_answers = pickle.load(f)
    with open('saved/tfidf_questions.npz', 'rb') as f:
        tfidf_questions = pickle.load(f)
    with open('saved/tfidf_vectorizer.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print('TF IDF loaded!')

    # Count Vectorizer
    print('loading Count...')
    with open('saved/count_answers.npz', 'rb') as f:
        count_answers = pickle.load(f)
    with open('saved/count_questions.npz', 'rb') as f:
        count_questions = pickle.load(f)
    with open('saved/count_vectorizer.pickle', 'rb') as f:
        count_vectorizer = pickle.load(f)
    print('Count loaded!')
    end = time()
    print('Time spent for loading - ', end - start)

    print('FastText scoring: ', scoring(fasttext_ques_matrix, fasttext_ans_matrix))
    print('Bert scoring: ', scoring(b_questions, b_answers))
    print('BM25 scoring: ', scoring(bm25_questions, bm25_answers))
    print('TF IDF scoring: ', scoring(tfidf_questions, tfidf_answers))
    print('Count scoring: ', scoring(count_questions, count_answers))

    while True:
        jaustalblin = input('Введите что-нибудь: fasttext, bert, bm25, tfidf, count: \n')
        if jaustalblin == 'fasttext':
            fasttext_search()

        elif jaustalblin == 'bert':
            bert_search()

        elif jaustalblin == 'bm25':
            bm25_tfidf_count_search(bm25_count_vectorizer, bm25_questions)

        elif jaustalblin == 'tfidf':
            start = time()
            bm25_tfidf_count_search(tfidf_vectorizer, tfidf_questions)
            end = time()
            print('Time spent for search - ', end - start)

        elif jaustalblin == 'count':
            start = time()
            bm25_tfidf_count_search(count_vectorizer, count_questions)
            end = time()
            print('Time spent for search - ', end - start)
        else:
            pass
