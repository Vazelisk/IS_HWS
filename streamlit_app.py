import streamlit as st
# from pymystem3 import Mystem
from string import punctuation
from scipy import sparse
import numpy as np
import pickle
from gensim.models import KeyedVectors
import torch
from transformers import AutoTokenizer, AutoModel
from time import time
import pymorphy2

# можно выбрать между майстемом и морфи (первый обрабатывает запрос долго)
morph = pymorphy2.MorphAnalyzer()
# m = Mystem()
punctuation += '...' + '—' + '…' + '«»'

with open('saved/answers_corpus.pickle', 'rb') as f:
    answers_corpus = pickle.load(f)

st.set_page_config(
    page_title="Lovely search",
    page_icon='💖',
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://raw.githubusercontent.com/Vazelisk/love_search/master/saved/background.png")
    }
    </style>
    """,
    unsafe_allow_html=True
)


def get_similarity(sparced_matrix, query_vec):
    scores = np.dot(sparced_matrix, query_vec.T)
    sorted_scores_indx = np.argsort(scores.toarray(), axis=0)[::-1]
    return list(np.array(answers_corpus)[sorted_scores_indx.ravel()][:25])


def fasttext_query_preprocessing(query):
    query = [word.lower().strip(punctuation).strip() for word in query.split()]
    # вариант майстема
    # query = m.lemmatize(' '.join([word for word in query]))
    # query = ' '.join([word for word in query])
    # tokens = [word for word in query.split() if word != '']
    # query = ' '.join([word for word in query.split() if word != ''])
    query = morph.parse(' '.join([word for word in query]))[0].normal_form

    return query.split()


def fasttext_search(query, fasttext_model, fasttext_ques_matrix):
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
        st.write('Time spent for search - ', end - start)
        return get_similarity(fasttext_ques_matrix, sparse.csr_matrix(query_vectors))


def bert_vectorizer(corpus, auto_model, auto_tokenizer):
    if corpus == '':
        return None
    else:
        corpus = [corpus[i:i + 3250] for i in range(0, len(corpus), 3250)]
        bert_vects = []
        for text in corpus:
            encoded_input = auto_tokenizer(text, padding=True, truncation=True, max_length=24, return_tensors='pt')
            encoded_input = encoded_input.to('cuda')
            with torch.no_grad():
                model_output = auto_model(**encoded_input)

            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings)
            for sentence in sentence_embeddings:
                bert_vects.append(sentence.cpu().numpy())
        torch.cuda.empty_cache()

        return sparse.csr_matrix(bert_vects)


def bert_search(query, auto_model, auto_tokenizer, b_questions):
    # start = time()
    query = [word.lower().strip(punctuation).strip() for word in query.split()]
    # end = time()
    # st.write(end - start)

    # start = time()
    query = morph.parse(' '.join([word for word in query]))[0].normal_form
    # query = m.lemmatize(' '.join([word for word in query]))
    # end = time()
    # st.write(end - start)
    # st.text(query)

    # start = time()
    # query = ' '.join([word for word in query])
    # end = time()
    # st.write(end - start)
    # st.text(query)

    query = ' '.join([word for word in query.split() if word != ''])
    # end = time()
    # st.write(end - start)
    # st.write(query)

    query_vec = bert_vectorizer(query, auto_model, auto_tokenizer)
    if query_vec is None:
        pass
    else:
        result = get_similarity(b_questions, query_vec)
        end = time()
        st.write('Time spent for search - ', end - start)
        return result


def tf_query_preprocessing(query, vectorizer):
    query = [word.lower().strip(punctuation).strip() for word in query.split()]
    query = morph.parse(' '.join([word for word in query]))[0].normal_form
    query = ' '.join([word for word in query.split() if word != ''])
    # вариант майстема
    # query = [word.lower().strip(punctuation).strip() for word in query.split()]
    # query = m.lemmatize(' '.join([word for word in query]))
    # query = ' '.join([word for word in query])
    # query = ' '.join([word for word in query.split() if word != ''])
    query_vec = vectorizer.transform([query])
    return query_vec


def bm25_tfidf_count_search(query, vectorizer, questions_matrix):
    start = time()
    query_vec = tf_query_preprocessing(query, vectorizer)
    end = time()
    result = get_similarity(questions_matrix, query_vec)
    st.write('Time spent for search - ', end - start)
    return result


def choose_model(option, query):
    if option == 'Count':
        count_result = bm25_tfidf_count_search(query, models['count_vectorizer'], models['count_questions'])
        return count_result

    if option == 'TF IDF':
        tfidf_result = bm25_tfidf_count_search(query, models['tfidf_vectorizer'], models['tfidf_questions'])
        return tfidf_result

    if option == 'BM25':
        bm25_result = bm25_tfidf_count_search(query, models['bm25_count_vectorizer'], models['bm25_questions'])
        return bm25_result

    if option == 'Bert':
        bert_result = bert_search(query, models['auto_model'], models['auto_tokenizer'], models['b_questions'])
        return bert_result

    if option == 'FastText':
        fasttext_result = fasttext_search(query, models['fasttext_model'], models['fasttext_ques_matrix'])
        return fasttext_result


@st.cache(allow_output_mutation=True)
def data_loader():
    models = {}
    with open('saved/count_questions.npz', 'rb') as f:
        count_questions = pickle.load(f)
        models['count_questions'] = count_questions
    with open('saved/count_vectorizer.pickle', 'rb') as f:
        count_vectorizer = pickle.load(f)
        models['count_vectorizer'] = count_vectorizer
    with open('saved/tfidf_questions.npz', 'rb') as f:
        tfidf_questions = pickle.load(f)
        models['tfidf_questions'] = tfidf_questions
    with open('saved/tfidf_vectorizer.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        models['tfidf_vectorizer'] = tfidf_vectorizer
    with open('saved/bm25_questions.npz', 'rb') as f:
        bm25_questions = pickle.load(f)
        models['bm25_questions'] = bm25_questions
    with open('saved/bm25_count_vectorizer.pickle', 'rb') as f:
        bm25_vectorizer = pickle.load(f)
        models['bm25_count_vectorizer'] = bm25_vectorizer
    auto_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    auto_model.to('cuda')
    b_questions = torch.load('saved/ques_bert.pt')
    models['auto_model'] = auto_model
    models['auto_tokenizer'] = auto_tokenizer
    models['b_questions'] = b_questions
    fasttext_model = KeyedVectors.load('saved/araneum_none_fasttextcbow_300_5_2018.model')
    fasttext_ques_matrix = sparse.load_npz('saved/fasttext_ques_matrix.npz')
    models['fasttext_model'] = fasttext_model
    models['fasttext_ques_matrix'] = fasttext_ques_matrix

    return models


if __name__ == '__main__':
    st.title('Lovely search')
    models = data_loader()
    with st.form(key='my_form'):
        option = st.selectbox('Which engine would you like to use?',
                              options=['Count', 'BM25', 'TF IDF', 'FastText', 'Bert'])
        query = st.text_input('Your question here:')
        total_answers = st.slider('How many answers do you want to see?', 0, 25, 1)
        submit_button = st.form_submit_button(label='Search')

    if submit_button:
        start = time()
        result = choose_model(option, query)

        if result is not None:
            for i in result[0:total_answers]:
                st.write('*', i)
            end = time()
            st.success('Total time spent: ' + str(end - start))
            st.balloons()
