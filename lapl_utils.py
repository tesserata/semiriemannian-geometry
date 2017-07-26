import numpy as np


def elementwise_square(matrices):
    square_taken = []
    for A in matrices:
        B = np.power(A, 2)
        square_taken.append(B)
    return square_taken


def elementwise_multiplication(matrices1, matrices2):
    multiplied = []
    for i in range(0, len(matrices1)):
        A1 = matrices1[i]
        A2 = matrices2[i]
        B = np.multiply(A1, A2)
        multiplied.append(B)
    return multiplied


def weight_by_squared_distance(matrices, inverse_distance_matrices):
    squared = elementwise_square(inverse_distance_matrices)
    weighted = elementwise_multiplication(matrices, squared)
    return weighted


def weight_by_distance(matrices, inverse_distance_matrices):
    weighted = elementwise_multiplication(matrices, inverse_distance_matrices)
    return weighted


def norm1(M):
    N = np.array(M).copy()
    N1 = []
    for A in N:
        total = np.float(np.sum(A))
        AN1 = np.divide(A, total)
        N1.append(AN1)
    return N1


def l2_only(list_of_matrices):
    M = np.array(list_of_matrices)
    n = M.shape[0]
    norm_l2 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i + 1, n):
            dist = M[i] - M[j]
            l2 = np.sqrt(np.sum(np.power(dist, 2)))
            norm_l2[i, j] = l2
            norm_l2[j, i] = l2
    return norm_l2


def laplacian(list_of_matrices):
    L_list = []
    for A in list_of_matrices:
        degrees = np.sum(A, 0)
        D = np.diag(degrees)
        L = D - A
        L_list.append(L)
    return np.array(L_list)


def produce_laplacian_datasets(list_of_matrices, list_of_inverted_weights):
    datasets = {}
    # original
    datasets["original_nonnormed"] = laplacian(list_of_matrices)
    datasets["original_n1normed"] = laplacian(norm1(list_of_matrices))
    # original by squared distances
    wsm = weight_by_squared_distance(list_of_matrices, list_of_inverted_weights)
    datasets["wbysqdist_nonnormed"] = laplacian(wsm)
    datasets["wbysqdist_n1normed"] = laplacian(norm1(wsm))
    # original by distances
    wm = weight_by_distance(list_of_matrices, list_of_inverted_weights)
    datasets["wbydist_nonnormed"] = laplacian(wm)
    datasets["wbydist_n1normed"] = laplacian(norm1(wm))
    # binarized
    binar = np.array(list_of_matrices).copy()
    binar[binar > 0] = 1
    datasets["binarized"] = laplacian(binar)
    return datasets


def normlaplacian(list_of_matrices):
    normL_list = []
    for A in list_of_matrices:
        degrees = np.sum(A, 0)
        D = np.diag(degrees)
        D_half_inv = np.diag(1. / np.sqrt(degrees))
        L = D - A
        L_normalized = np.dot(np.dot(D_half_inv, L), D_half_inv)
        normL_list.append(L_normalized)
    return np.array(normL_list)


def produce_normlaplacian_datasets(list_of_matrices, list_of_inverted_weights):
    datasets = {}
    # original
    datasets["original_nonnormed"] = normlaplacian(list_of_matrices)
    # original by squared distances
    wsm = weight_by_squared_distance(list_of_matrices, list_of_inverted_weights)
    datasets["wbysqdist_nonnormed"] = normlaplacian(wsm)
    # original by distances
    wm = weight_by_distance(list_of_matrices, list_of_inverted_weights)
    datasets["wbydist_nonnormed"] = normlaplacian(wm)
    # binarized
    binar = np.array(list_of_matrices).copy()
    binar[binar > 0] = 1
    datasets["binarized"] = normlaplacian(binar)
    return datasets
