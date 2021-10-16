# принимает спарс матрицы
import numpy as np


def scoring(q_matrix, a_matrix):
    # q_matrix = np.delete(q_matrix., np.s_[5000:], 0)
    # a_matrix = np.delete(a_matrix.toarray(), np.s_[5000:], 0)

    scoring_matrix = np.dot(q_matrix.toarray(), a_matrix.toarray().T)
    score = 0
    for ind, line in enumerate(scoring_matrix):
        sorted_scores_indx = np.argsort(line, axis=0)[::-1]
        sorted_scores_indx = [sorted_scores_indx.ravel()][0][:5]
        if ind in sorted_scores_indx:
            score += 1

    return score / q_matrix.shape[0]

if __name__ == '__main__':
    pass
