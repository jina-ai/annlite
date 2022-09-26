import numpy as np
import pytest
from docarray import Document, DocumentArray


def test_save_load(tmpfile):
    save_da = DocumentArray(
        storage='annlite', config={'n_dim': 768, 'data_path': tmpfile}
    )
    for i in range(100):
        save_da.append(Document(id=str(i), embedding=np.random.rand(768)))

    load_da = DocumentArray(
        storage='annlite', config={'n_dim': 768, 'data_path': tmpfile}
    )
    assert len(load_da) == len(save_da)

    for i in range(100, 120):
        load_da.append(Document(id=str(i), embedding=np.random.rand(768)))
    assert len(load_da) == 120
