import numpy as np
from docarray import Document, DocumentArray


def test_find():
    nrof_docs = 1000
    num_candidates = 100

    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
        },
    )

    with annlite_doc:
        annlite_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ],
        )

    np_query = np.array([2, 1, 3])

    annlite_doc.find(np_query, limit=10, num_candidates=num_candidates)
