import os

import numpy as np
import pytest

from annlite.core.index.hnsw import HnswIndex

n_examples = 100
n_features = 10


@pytest.fixture
def build_data():
    Xt = np.random.random((n_examples, n_features)).astype(np.float32)
    return Xt


@pytest.fixture
def build_hnsw(build_data):
    Xt = build_data
    hnsw = HnswIndex(dim=n_features)
    hnsw.add_with_ids(x=Xt, ids=list(range(len(Xt))))
    return hnsw


def test_save_and_load(tmpdir, build_hnsw):
    hnsw = build_hnsw
    hnsw.dump(os.path.join(tmpdir, 'hnsw.pkl'))
    assert os.path.exists(os.path.join(tmpdir, 'hnsw.pkl')) is True

    hnsw_ = HnswIndex(dim=n_features, index_file=os.path.join(tmpdir, 'hnsw.pkl'))
    assert hnsw_.size == hnsw.size


def test_loading_from_wrong_path(tmpfile):
    with pytest.raises(FileNotFoundError):
        HnswIndex(dim=n_features, index_file=os.path.join(tmpfile, 'hnsw_wrong.pkl'))
