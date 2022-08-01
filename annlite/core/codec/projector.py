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
    :param is_incremental: whether to use incremental projector, especially useful when
            data size is too large to be loaded into memory at once.
    :param batch_size: batch size used for incremental projector.
    """

    def __init__(
        self,
        n_components: int = 128,
        whiten: Optional[bool] = False,
        svd_solver: Optional[str] = 'auto',
        is_incremental: Optional[bool] = False,
        batch_size: Optional[int] = 512,
    ):
        super(ProjectorCodec, self).__init__(require_train=True)
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver

        self._is_trained = False

        self.is_incremental = is_incremental
        if not self.is_incremental:
            from sklearn.decomposition import PCA

            # fix the random seed to make sure that we can get the same result in each
            # function call
            self.pca = PCA(n_components=self.n_components, random_state=1234)
        else:
            from sklearn.decomposition import IncrementalPCA

            self.batch_size = batch_size
            self.pca = IncrementalPCA(
                n_components=self.n_components, batch_size=self.batch_size
            )

    def fit(self, x: 'np.ndarray'):
        """Train projector model

        :param x: Training vectors with shape=(N, D)
        """
        assert x.dtype == np.float32
        assert x.ndim == 2
        assert (
            x.shape[0] > self.n_components
        ), 'number of input data must be larger than or equal to n_components'

        self.pca.fit(x)
        self._is_trained = True

    def partial_fit(self, x: 'np.ndarray'):
        """Given a batch of training vectors, update the internal projector.
        This method is specially designed to be used when data does not fit in memory.

        :param x: Training vectors with shape=(N, D)
        """
        assert x.dtype == np.float32
        assert x.ndim == 2
        assert (
            x.shape[0] > self.n_components
        ), 'number of input data must be larger than or equal to n_components'
        assert (
            len(x) % self.batch_size == 0
        ), 'number of input data must be divided by batch size'

        for i in range(0, len(x), self.batch_size):
            self.pca.partial_fit(x[i : i + self.batch_size])
        self._is_trained = True

    def encode(self, x: 'np.ndarray'):
        """Encode input vectors using projector.

        :param x: Input vectors with shape=(N, D) and dtype=np.float32.
        :return: np.ndarray: transformed vectors using projector.
        """
        assert x.dtype == np.float32
        assert x.ndim == 2

        return self.pca.transform(x)

    def decode(self, x: 'np.ndarray'):
        """Given transformed vectors, reconstruct original D-dimensional vectors
        approximately.

        :param x: vectors with shape=(N, self.n_components).
        :return: Reconstructed vectors with shape=(N, D) and dtype=np.float32
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
        assert self.is_incremental is True, (
            'Per-feature empirical variance only be available when incremental '
            'project is used'
        )
        self._check_trained()
        return self.pca.var_
