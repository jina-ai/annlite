import pytest
import numpy as np
from pqlite.core.codec.pq import PQCodec
from pqlite.utils.asymmetric_distance import precompute_adc_table, dist_pqcodes_to_codebooks

n_examples = 2000
n_features = 128  # dimensionality / number of features
n_queries = 5
n_cells = 10
n_clusters = 256
n_subvectors = 32
top_k = 100

@pytest.fixture
def build_data():
    Xt = np.random.random((n_examples, n_features)).astype(
        np.float32
    )
    return Xt

@pytest.fixture
def build_pq_codec(build_data):
    Xt = build_data
    pq_codec = PQCodec(d_vector = n_features)
    pq_codec.fit(Xt)
    return pq_codec


def test_pq_adc_table_computation(build_pq_codec, build_data):

    def numpy_adc_table(query, n_subvectors, n_clusters, d_subvector, codebooks):
        dtable = np.empty((n_subvectors, n_clusters), dtype=np.float32)
        for m in range(n_subvectors):
            query_sub = query[m * d_subvector: (m + 1) * d_subvector]
            dtable[m, :] = np.linalg.norm(codebooks[m] - query_sub, axis=1) ** 2

        return dtable

    pq_codec = build_pq_codec
    query = build_data[0]

    np_distance_table = numpy_adc_table(query,
                                        pq_codec.n_subvectors,
                                        pq_codec.n_clusters,
                                        pq_codec.d_subvector,
                                        pq_codec.codebooks)

    distance_table_cy = precompute_adc_table(query,
                                             pq_codec.d_subvector,
                                             pq_codec.n_clusters,
                                             pq_codec.codebooks)

    np_distance_table_cy = np.asarray(distance_table_cy)
    np.testing.assert_array_almost_equal(np_distance_table, np_distance_table_cy, decimal=5)


def test_pq_adc_table_computation_interface(build_pq_codec, build_data):
    pq_codec = build_pq_codec
    query = build_data[0]

    np_distance_table = pq_codec.precompute_adc(query).dtable

    distance_table_cy = precompute_adc_table(query,
                                             pq_codec.d_subvector,
                                             pq_codec.n_clusters,
                                             pq_codec.codebooks)

    np_distance_table_cy = np.asarray(distance_table_cy)

    np.testing.assert_array_almost_equal(np_distance_table, np_distance_table_cy, decimal=5)

