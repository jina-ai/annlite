import numpy as np
import pytest
from docarray import Document, DocumentArray


def test_save_load(tmpfile):
    N = 100

    save_da = DocumentArray(
        storage='annlite', config={'n_dim': 768, 'data_path': tmpfile}
    )
    for i in range(N):
        save_da.append(Document(id=str(i), embedding=np.random.rand(768)))

    # need release the resource
    save_da._annlite.close()

    load_da = DocumentArray(
        storage='annlite', config={'n_dim': 768, 'data_path': tmpfile}
    )
    load_da._annlite.restore()
    assert len(load_da) == N

    for i in range(N, N + N):
        load_da.append(Document(id=str(i), embedding=np.random.rand(768)))
    assert len(load_da) == N + N
