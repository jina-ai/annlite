import os

import numpy as np
import pytest

from annlite.core.index.hnsw import HnswIndex

n_examples = 2000
n_features = 512


@pytest.fixture
def build_data():
    Xt = np.random.random((n_examples, n_features)).astype(np.float32)
    return Xt


@pytest.fixture
def build_hnsw(build_data):
    Xt = build_data
    hnsw = HnswIndex(dim=512)
    hnsw.add_with_ids(x=Xt, ids=list(range(len(Xt))))
    return hnsw


def test_save(tmpdir, build_hnsw):
    hnsw = build_hnsw
    hnsw.save_index(os.path.join(tmpdir, 'hnsw.pkl'))

    assert os.path.exists(os.path.join(tmpdir, 'hnsw.pkl')) == True


def test_load(tmpdir, build_hnsw):
    hnsw = build_hnsw
    hnsw.save_index(os.path.join(tmpdir, 'hnsw.pkl'))

    hnsw_ = HnswIndex(dim=512, path_to_load=os.path.join(tmpdir, 'hnsw.pkl'))
    assert hnsw_.size == hnsw.size


def test_loading_from_wrong_path(tmpdir):
    hnsw_wrong = HnswIndex(dim=512, path_to_load=os.path.join(tmpdir, 'hnsw_wrong.pkl'))

    assert hnsw_wrong.size == 0
