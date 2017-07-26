from scipy.linalg import svd, eigh, eigvals
import warnings
import numpy as np
from pyriemann.utils.distance import distance_riemann, distance_euclid
from pyriemann.utils.base import logm


def _eigh_list(x, rank):
    D_x, V_x = eigh(x)
    D_x, V_x = D_x[-1:-rank - 1:-1], V_x[:, -1:- rank - 1:-1]
    return (D_x, V_x)


def _matrix_compute(D_list, V_list, i, j, rank):
    D_a = D_list[i]
    V_a = V_list[i]
    D_b = D_list[j]
    V_b = V_list[j]
    O_a, sigma, O_b = svd(np.dot(V_a.T, V_b), lapack_driver='gesvd')
    O_b = O_b.T  # add transpose missed in scipy
    if (np.abs(sigma).max() - 1) <= 1e-15:
        warnings.warn('cosines are too large')
    sigma[sigma > 1] = 1  # to deal with floating-point arithmetic
    sigma[sigma < -1] = -1
    theta = np.arccos(sigma)

    R_a_sq = np.dot(np.dot(O_a.T, np.diag(D_a)), O_a)
    R_b_sq = np.dot(np.dot(O_b.T, np.diag(D_b)), O_b)

    length_sq = np.sum(theta**2) + rank * np.sum(np.log(eigvals(R_a_sq, R_b_sq))**2)
    d = length_sq**0.5
    return d


def spsd_distance_matrix(X, aggregate=True, k=None):
    """
    Accepts a list (or an array) of matrices (does not check for positive semidefiniteness - make sure they are!)
    Returns a matrix of pairwise distances between these matrices
    """

    # compute minimum rank of the matrices from the array:
    X = np.array(X)
    rank = np.min([np.linalg.matrix_rank(x) for x in X])
    if k is None:
        k = rank  # for the backward compability

    V_list = []
    D_list = []

    for x in X:
        D_x, V_x = eigh(x)
        D_x, V_x = D_x[-1:-rank - 1:-1], V_x[:, -1:- rank - 1:-1]
        D_list.append(D_x)
        V_list.append(V_x)

    dim = X.shape[0]
    angles_matrix = np.zeros((dim, dim))
    subriemann_matrix = np.zeros((dim, dim))
    for i in range(0, dim):
        for j in range(i + 1, dim):  # does not compute the diagonal elements (are of order e^(-6))
            S_x = D_list[i]
            V_x = V_list[i]
            S_z = D_list[j]
            V_z = V_list[j]

            O_x, sigma, O_z = svd(np.dot(V_x.T, V_z), lapack_driver='gesvd')
            O_z = O_z.T  # add transpose missed in scipy
            if (np.abs(sigma).max() - 1) <= 1e-15:
                warnings.warn('cosines are too large')
            sigma[sigma > 1] = 1  # to deal with floating-point arithmetic
            sigma[sigma < -1] = -1
            theta = np.arccos(sigma)

            R_x_sq = np.dot(np.dot(O_x.T, np.diag(S_x)), O_x)
            R_z_sq = np.dot(np.dot(O_z.T, np.diag(S_z)), O_z)

            angles_matrix[i, j] = np.sum(theta**2)**0.5
            angles_matrix[j, i] = np.sum(theta**2)**0.5

            subriemann_matrix[i, j] = np.sum(np.log(eigvals(R_x_sq, R_z_sq))**2)**0.5
            subriemann_matrix[j, i] = np.sum(np.log(eigvals(R_x_sq, R_z_sq))**2)**0.5
    dist_matrix = angles_matrix + k * subriemann_matrix

    return dist_matrix if aggregate else [angles_matrix, subriemann_matrix]


def exponent_kernel(distance_matrix, alpha=1):
    kernel = np.exp(- alpha * distance_matrix)
    return kernel


def gaussian_kernel(distance_matrix, alpha=1):
    kernel = np.exp(- alpha * distance_matrix ** 2)
    return kernel


def distance_matrix(X, metric):
    """
    Accepts a list (or an array) of matrices
    (does not check for positive semidefiniteness - make sure they are!)
    Returns a matrix of pairwise distances between these matrices
    """
    dim = X.shape[0]
    dist_matrix = np.zeros((dim, dim))
    if metric == 'spsd':
        X = np.array(X)
        rank = np.min([np.linalg.matrix_rank(x) for x in X])

        V_list = []
        D_list = []

        for x in X:
            D_x, V_x = eigh(x)
            D_x, V_x = D_x[-1:-rank - 1:-1], V_x[:, -1:- rank - 1:-1]
            D_list.append(D_x)
            V_list.append(V_x)

        for i in range(0, dim):
            for j in range(i + 1, dim):  # does not compute the diagonal elements (are of order e^(-6))
                D_a = D_list[i]
                V_a = V_list[i]
                D_b = D_list[j]
                V_b = V_list[j]

                O_a, sigma, O_b = svd(np.dot(V_a.T, V_b), lapack_driver='gesvd')
                O_b = O_b.T  # add transpose missed in scipy
                if (np.abs(sigma).max() - 1) <= 1e-15:
                    warnings.warn('cosines are too large')
                sigma[sigma > 1] = 1  # to deal with floating-point arithmetic
                sigma[sigma < -1] = -1
                theta = np.arccos(sigma)

                R_a_sq = np.dot(np.dot(O_a.T, np.diag(D_a)), O_a)
                R_b_sq = np.dot(np.dot(O_b.T, np.diag(D_b)), O_b)

                length_sq = np.sum(theta**2) + rank * np.sum(np.log(eigvals(R_a_sq, R_b_sq))**2)
                d = length_sq**0.5

                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    elif metric == 'spd':
        for i in range(dim):
            for j in range(i + 1, dim):
                X_i = X[i] + np.identity(X[i].shape[0]) * 1e-03
                X_j = X[j] + np.identity(X[j].shape[0]) * 1e-03
                d = distance_riemann(X_i, X_j)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    elif metric == 'euclid':
        for i in range(dim):
            for j in range(i + 1, dim):  # does not compute the diagonal elements (are of order e^(-6))
                d = distance_euclid(X[i], X[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    return dist_matrix


def get_log_euclid(X):
    """
    Accepts a list (or an array) of SPD matrices
    Returns a matrix of pairwise distances between these matrices
    """

    X = np.array([logm(x) for x in X])

    dim = X.shape[0]
    dist_matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):  # does not compute the diagonal elements (are of order e^(-6))
            d = distance_euclid(X[i], X[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix
