from pymystem3 import Mystem
from string import punctuation
import nltk
from nltk.corpus import stopwords
import re
from scipy import sparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from gensim.models import KeyedVectors
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from time import time

m = Mystem()
nltk.download('stopwords')
stopword = stopwords.words('russian')
punctuation += '...' + '—' + '…' + '«»'


def first_processing(file):
    with open(file, 'r', encoding='utf-8') as f:
        # установите размер корпуса в предложениях (для 50к нужно более 10гб памяти)
        corpus = list(f)[:10000]

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


def fasttext_get_matrix(texts, fasttext_model):
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


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def bert_vectorizer(corpus, auto_model, auto_tokenizer):
    # на 3000 файлов требуется ~7gb видеопамяти. Устанавливайте значение ниже, если памяти меньше
    corpus = [corpus[i:i + 3000] for i in range(0, len(corpus), 3000)]
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


def tfidf_vectorization(ans_cleared_corpus, que_cleared_corpus):
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf_answers = vectorizer.fit_transform(ans_cleared_corpus)
    tfidf_questions = vectorizer.transform(que_cleared_corpus)

    return tfidf_answers, tfidf_questions, vectorizer


def count_vectorization(ans_cleared_corpus, que_cleared_corpus):
    vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    count_answers = vectorizer.fit_transform(ans_cleared_corpus)
    count_questions = vectorizer.transform(que_cleared_corpus)

    return count_answers, count_questions, vectorizer


def create_all():
    # для создания корпусов
    start = time()
    print('creating corpuses')
    answers_corpus, questions_corpus = first_processing('saved/questions_about_love.jsonl')
    answers_corpus, questions_corpus, cleared_answers_corpus, cleared_questions_corpus = get_cleared_corpus(
        answers_corpus, questions_corpus)
    with open('saved/answers_corpus.pickle', 'wb') as f:
        pickle.dump(answers_corpus, f)
    with open('saved/questions_corpus.pickle', 'wb') as f:
        pickle.dump(questions_corpus, f)
    with open('saved/cleared_answers_corpus.pickle', 'wb') as f:
        pickle.dump(cleared_answers_corpus, f)
    with open('saved/cleared_questions_corpus.pickle', 'wb') as f:
        pickle.dump(cleared_questions_corpus, f)
    end = time()
    print('corpuses created', end - start)

    # fasttext module
    start = time()
    print('creating fasttext matrixes')
    fasttext_model = KeyedVectors.load('saved/araneum_none_fasttextcbow_300_5_2018.model')
    fasttext_ans_matrix = fasttext_get_matrix(cleared_answers_corpus, fasttext_model)
    fasttext_ques_matrix = fasttext_get_matrix(cleared_questions_corpus, fasttext_model)
    sparse.save_npz('saved/fasttext_ans_matrix.npz', fasttext_ans_matrix)
    sparse.save_npz('saved/fasttext_ques_matrix.npz', fasttext_ques_matrix)
    end = time()
    print('created fasttext matrixes', end - start)

    # bert
    start = time()
    print('creating bert matrixes')
    auto_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_model.to('cuda')
    torch.save(bert_vectorizer(cleared_questions_corpus, auto_model, auto_tokenizer), 'saved/ques_bert.pt')
    torch.save(bert_vectorizer(cleared_answers_corpus, auto_model, auto_tokenizer), 'saved/ans_bert.pt')
    end = time()
    print('created bert matrixes', end - start)

    # BM25
    start = time()
    print('creating bm25 matrixes')
    bm25_answers, bm25_questions, bm25_count_vectorizer = bm25_vectorization(cleared_answers_corpus,
                                                                             cleared_questions_corpus)
    with open('saved/bm25_answers.npz', 'wb') as f:
        pickle.dump(bm25_answers, f)
    with open('saved/bm25_questions.npz', 'wb') as f:
        pickle.dump(bm25_questions, f)
    with open('saved/bm25_count_vectorizer.pickle', 'wb') as f:
        pickle.dump(bm25_count_vectorizer, f)
    end = time()
    print('created bm25 matrixes', end - start)

    # TF IDF
    start = time()
    print('creating tfidf matrixes')
    tfidf_answers, tfidf_questions, tfidf_vectorizer = tfidf_vectorization(cleared_answers_corpus,
                                                                           cleared_questions_corpus)
    with open('saved/tfidf_answers.npz', 'wb') as f:
        pickle.dump(tfidf_answers, f)
    with open('saved/tfidf_questions.npz', 'wb') as f:
        pickle.dump(tfidf_questions, f)
    with open('saved/tfidf_vectorizer.pickle', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    end = time()
    print('created tfidf matrixes', end - start)

    # Count
    start = time()
    print('creating count matrixes')
    count_answers, count_questions, count_vectorizer = count_vectorization(cleared_answers_corpus,
                                                                           cleared_questions_corpus)
    with open('saved/count_answers.npz', 'wb') as f:
        pickle.dump(count_answers, f)
    with open('saved/count_questions.npz', 'wb') as f:
        pickle.dump(count_questions, f)
    with open('saved/count_vectorizer.pickle', 'wb') as f:
        pickle.dump(count_vectorizer, f)
    end = time()
    print('created count matrixes', end - start)


if __name__ == '__main__':
    create_all()
