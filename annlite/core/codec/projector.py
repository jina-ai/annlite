from typing import Optional

import numpy as np

from .base import BaseCodec


class ProjectorCodec(BaseCodec):
    """Implementation of Projector.

    :param n_components: number of components to keep.
    :param whiten: when True (False by default) the components_ vectors are multiplied
                    by the square root of n_samples and then divided by the singular
                    values to ensure uncorrelated outputs with unit component-wise variances.
    :param svd_solver:
            If auto: The solver is selected by a default policy based on X.shape and
            n_components: if the input data is larger than 500x500 and the number of
            components to extract is lower than 80% of the smallest dimension of the
            data, then the more efficient ‘randomized’ method is enabled. Otherwise
            the exact full SVD is computed and optionally truncated afterwards.

            If full: run exact full SVD calling the standard LAPACK solver via scipy.
            linalg.svd and select the components by postprocessing.

            If arpack: run SVD truncated to n_components calling ARPACK solver via
            scipy.sparse.linalg.svds. It requires strictly 0 < n_components < min(X.shape).
    """

    def __init__(
        self,
        dim: int,
        n_components: int = 128,
        whiten: Optional[bool] = False,
        svd_solver: Optional[str] = 'auto',
    ):
        super(ProjectorCodec, self).__init__(require_train=True)
        self.dim = dim
        self.n_components = n_components
        assert self.dim >= self.n_components, (
            f'the dimension after projector should be less than original dimension, got '
            f'original dimension: {self.dim} and projector dimension: {self.n_components}'
        )

        self.whiten = whiten
        self.svd_solver = svd_solver

        self.pca = None

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                self.dim,
                self.n_components,
                self.whiten,
                self.svd_solver,
            )
        )

    def fit(self, x: 'np.ndarray'):
        """Train projector model

        :param x: Training vectors with shape=(N, D)
        """
        assert x.ndim == 2
        assert (
            x.shape[1] == self.dim,
        ), 'dimension of input data must be equal to "dim"'
        assert (
            x.shape[0] > self.n_components
        ), 'number of input data must be larger than or equal to n_components'

        if self.pca is None:
            from sklearn.decomposition import PCA

            self.pca = PCA(
                n_components=self.n_components,
                whiten=self.whiten,
                svd_solver=self.svd_solver,
            )

        self.pca.fit(x)
        self._is_trained = True

    def partial_fit(self, x: 'np.ndarray'):
        """Given a batch of training vectors, update the internal projector.
        This method is specially designed to be used when data does not fit in memory.

        :param x: Training vectors with shape=(N, D)
        """

        assert x.ndim == 2
        assert x.shape[1] == self.dim, 'dimension of input data must be equal to "dim"'
        assert (
            x.shape[0] > self.n_components
        ), 'number of input data must be larger than or equal to n_components'

        if self.pca is None:
            from sklearn.decomposition import IncrementalPCA

            self.pca = IncrementalPCA(
                n_components=self.n_components,
                whiten=self.whiten,
            )

        self.pca.partial_fit(x)
        self._is_trained = True

    def encode(self, x: 'np.ndarray'):
        """Encode input vectors using projector.

        :param x: Input vectors with shape=(N, D)
        :return: np.ndarray: transformed vectors using projector.
        """
        assert x.ndim == 2
        assert x.shape[1] == self.dim, 'dimension of input data must be equal to "dim"'

        return self.pca.transform(x)

    def decode(self, x: 'np.ndarray'):
        """Given transformed vectors, reconstruct original D-dimensional vectors
        approximately.

        :param x: vectors with shape=(N, self.n_components).
        :return: Reconstructed vectors with shape=(N, D)
        """
        assert x.ndim == 2
        assert x.shape[1] == self.n_components

        return self.pca.inverse_transform(x)

    @property
    def components(self):
        """Principal axes in feature space, representing the directions of maximum
        variance in the data.
        """
        self._check_trained()
        return self.pca.components_

    @property
    def explained_variance_ratio(self):
        """Percentage of variance explained by each of the selected components."""
        self._check_trained()
        return self.pca.explained_variance_ratio_

    @property
    def mean(self):
        """Per-feature empirical mean."""
        self._check_trained()
        return self.pca.mean_

    @property
    def var(self):
        """Per-feature empirical variance"""
        self._check_trained()
        return self.pca.var_
