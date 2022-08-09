from typing import Tuple

import numpy as np


def l2_normalize(x: 'np.ndarray', eps: float = np.finfo(np.float32).eps):
    """Scale input vectors individually to unit norm.

    :param x: The data to normalize
    :param eps: a small jitter to avoid divde by zero
    :return: Normalized input X
    """

    norms = np.einsum('ij,ij->i', x, x)
    np.sqrt(norms, norms)
    constant_mask = norms < 10 * eps
    norms[constant_mask] = 1.0
    return x / norms[:, np.newaxis]


def cosine(
    x_mat: 'np.ndarray', y_mat: 'np.ndarray', eps: float = np.finfo(np.float32).eps
) -> 'np.ndarray':
    """Cosine distance between each row in x_mat and each row in y_mat.

    :param x_mat: np.ndarray with ndim=2
    :param y_mat: np.ndarray with ndim=2
    :param eps: a small jitter to avoid divde by zero
    :return: np.ndarray  with ndim=2
    """
    return 1 - np.clip(
        (np.dot(x_mat, y_mat.T) + eps)
        / (
            np.outer(np.linalg.norm(x_mat, axis=1), np.linalg.norm(y_mat, axis=1)) + eps
        ),
        -1,
        1,
    )


def sqeuclidean(x_mat: 'np.ndarray', y_mat: 'np.ndarray') -> 'np.ndarray':
    """Squared Euclidean distance between each row in x_mat and each row in y_mat.
    :param x_mat: np.ndarray with ndim=2
    :param y_mat: np.ndarray with ndim=2
    :return: np.ndarray with ndim=2
    """
    return (
        np.sum(y_mat**2, axis=1)
        + np.sum(x_mat**2, axis=1)[:, np.newaxis]
        - 2 * np.dot(x_mat, y_mat.T)
    )


def euclidean(x_mat: 'np.ndarray', y_mat: 'np.ndarray') -> 'np.ndarray':
    """Euclidean distance between each row in x_mat and each row in y_mat.

    :param x_mat:  scipy.sparse like array with ndim=2
    :param y_mat:  scipy.sparse like array with ndim=2
    :return: np.ndarray  with ndim=2
    """
    return np.sqrt(sqeuclidean(x_mat, y_mat))


def pdist(
    x_mat: 'np.ndarray',
    metric: str,
) -> 'np.ndarray':
    """Computes Pairwise distances between observations in n-dimensional space.

    :param x_mat: Union['np.ndarray','scipy.sparse.csr_matrix', 'scipy.sparse.coo_matrix'] of ndim 2
    :param metric: string describing the metric type
    :return: np.ndarray of ndim 2
    """
    return cdist(x_mat, x_mat, metric)


def cdist(x_mat: 'np.ndarray', y_mat: 'np.ndarray', metric: str) -> 'np.ndarray':
    """Computes the pairwise distance between each row of X and each row on Y according to `metric`.
    - Let `n_x = x_mat.shape[0]`
    - Let `n_y = y_mat.shape[0]`
    - Returns a matrix `dist` of shape `(n_x, n_y)` with `dist[i,j] = metric(x_mat[i], y_mat[j])`.
    :param x_mat: numpy or scipy array of ndim 2
    :param y_mat: numpy or scipy array of ndim 2
    :param metric: string describing the metric type
    :return: np.ndarray of ndim 2
    """
    dists = {'cosine': cosine, 'sqeuclidean': sqeuclidean, 'euclidean': euclidean}[
        metric
    ](x_mat, y_mat)

    return dists


def top_k(
    values: 'np.ndarray', k: int, descending: bool = False
) -> Tuple['np.ndarray', 'np.ndarray']:
    """Finds values and indices of the k largest entries for the last dimension.

    :param values: array of distances
    :param k: number of values to retrieve
    :param descending: find top k biggest values
    :return: indices and distances
    """
    if descending:
        values = -values

    if k >= values.shape[1]:
        idx = values.argsort(axis=1)[:, :k]
        values = np.take_along_axis(values, idx, axis=1)
    else:
        idx_ps = values.argpartition(kth=k, axis=1)[:, :k]
        values = np.take_along_axis(values, idx_ps, axis=1)
        idx_fs = values.argsort(axis=1)
        idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
        values = np.take_along_axis(values, idx_fs, axis=1)

    if descending:
        values = -values

    return values, idx
