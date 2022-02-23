import numpy as np
import pytest

from annlite.core.codec.pq import PQCodec

n_examples = 1000
n_features = 128
n_queries = 5
n_cells = 10
n_clusters = 256
n_subvectors = 32
d_subvector = int(n_features / n_subvectors)
top_k = 100


@pytest.fixture
def build_data():
    Xt = np.random.random((n_examples, n_features)).astype(np.float32)
    return Xt


@pytest.fixture
def build_pq_codec(build_data):
    Xt = build_data
    pq_codec = PQCodec(dim=n_features, n_subvectors=n_subvectors, n_clusters=n_clusters)
    pq_codec.fit(Xt)
    return pq_codec


def minibatch_generator(Xtr, batch_size):
    num = 0
    pos_begin_batch = 0
    n_examples = len(Xtr)

    while True:
        Xtr_batch = Xtr[pos_begin_batch : pos_begin_batch + batch_size]
        yield Xtr_batch

        num += len(Xtr_batch)
        pos_begin_batch += batch_size

        if num + batch_size >= n_examples:
            break


@pytest.fixture
def build_pq_codec_online(build_data):
    Xt = build_data
    pq_codec_minibatch = PQCodec(
        dim=n_features, n_subvectors=n_subvectors, n_clusters=n_clusters
    )
    n_epochs = 3
    n_batch = 300

    for i in range(n_epochs):
        minibatch_generator_ = minibatch_generator(Xt, n_batch)

        for batch in minibatch_generator_:
            pq_codec_minibatch.partial_fit(batch)

    pq_codec_minibatch.build_codebook()
    return pq_codec_minibatch


def test_partial_and_total_fit_same_codebook_shape(
    build_pq_codec, build_pq_codec_online
):
    pq_codec = build_pq_codec
    pq_codec_minibatch = build_pq_codec_online
    assert pq_codec.codebooks.shape == pq_codec_minibatch.codebooks.shape
