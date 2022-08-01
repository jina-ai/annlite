import numpy as np
import pytest

from annlite.core.codec.projector import ProjectorCodec

n_examples = 1000
n_features = 512
n_components = 128
batch_size = 200


@pytest.fixture
def build_data():
    Xt = np.random.random((n_examples, n_features)).astype(np.float32)
    return Xt


@pytest.fixture
def build_projector(build_data):
    Xt = build_data

    projector_list = []
    for is_incremental in [True, False]:
        projector = ProjectorCodec(
            n_components=n_components,
            is_incremental=is_incremental,
            batch_size=batch_size,
        )
        if is_incremental:
            for i in range(0, len(Xt), batch_size):
                projector.partial_fit(Xt[i : i + batch_size])
                i += batch_size
        else:
            projector.fit(Xt)
        projector_list.append(projector)
    return projector_list


def test_encode_decode(build_data, build_projector):
    Xt = build_data

    projector_list = build_projector

    for projector in projector_list:
        transformed_vecs = projector.encode(Xt)
        assert transformed_vecs.shape == (Xt.shape[0], n_components)

        original_vecs = projector.decode(transformed_vecs)
        assert original_vecs.shape == Xt.shape


@pytest.mark.parametrize('insert_size', [10, 190, 390])
def test_wrong_insert_size(build_data, insert_size):
    Xt = build_data[:insert_size]

    projector = ProjectorCodec(
        n_components=n_components, is_incremental=True, batch_size=batch_size
    )
    with pytest.raises(Exception):
        projector.partial_fit(Xt)
