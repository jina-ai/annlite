import time
from datetime import date

import numpy as np
import pandas as pd
from docarray import Document, DocumentArray
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from utils import evaluate

from pqlite import PQLite
from pqlite.math import cdist
from pqlite.math import top_k as _top_k

# N = 100_000 # number of data points
Nt = 100_020
Nq = 1
D = 128  # dimentionality / number of features
top_k = 10
n_cells = 64
n_subvectors = 64
n_queries = 1000

# 2,000 128-dim vectors for training
np.random.seed(123)
Xtr, Xte = train_test_split(
    make_blobs(n_samples=Nt, n_features=D)[0].astype(np.float32), test_size=20
)
print(f'Xtr: {Xtr.shape} vs Xte: {Xte.shape}')


def get_documents(nr=10, index_start=0, embeddings=None):
    for i in range(index_start, nr + index_start):
        d = Document()
        d.id = f'{i}'  # to test it supports non-int ids
        d.embedding = embeddings[i - index_start]
        yield d


precision_per_query = []
recall_per_query = []
results = []

for n_cells in [1, 4, 8]:
    for n_subvectors in [64, 128]:

        pq = PQLite(
            dim=D, metric='euclidean', n_cells=n_cells, n_subvectors=n_subvectors
        )

        t0 = time.time()
        pq.train(Xtr[:20480])
        train_time = abs(time.time() - t0)

        t0 = time.time()
        pq.index(DocumentArray(get_documents(len(Xtr), embeddings=Xtr)))
        index_time = abs(t0 - time.time())

        dists = cdist(Xte, Xtr, metric='euclidean')
        true_dists, true_ids = _top_k(dists, top_k, descending=False)

        docs = DocumentArray(get_documents(len(Xte), embeddings=Xte))
        t0 = time.time()
        pq.search(docs, limit=top_k)
        query_time = abs(t0 - time.time())

        pq_ids = []
        for doc in docs:
            pq_ids.append([m.id for m in doc.matches])

        recall, precision = evaluate(pq_ids, true_ids, top_k)

        results_dict = {
            'precision': precision,
            'recall': recall,
            'train_time': train_time,
            'index_time': index_time,
            'query_time': query_time,
            'query_qps': len(Xte) / query_time,
            'index_qps': len(Xtr) / index_time,
            'indexer_hyperparams': {'n_cells': n_cells, 'n_subvectors': n_subvectors},
        }
        print(results_dict)

        results.append(results_dict)
        pq.clear()
        pq.close()

today = date.today()
results_df = pd.DataFrame(results)
results_df.sort_values('recall', ascending=False)
results_df.to_csv(f'bench-results-{today.strftime("%b-%d-%Y")}.csv')
