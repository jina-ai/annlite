import pqlite
pqlite.__path__
import time

import jina
from jina.math.distance import cdist

import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import random
import numpy as np
from pqlite import PQLite


def _precision(predicted, relevant, eval_at):
    """
    fraction of retrieved documents that are relevant to the query
    """
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(set(predicted_at_k).intersection(set(relevant)))

    return n_predicted_and_relevant / len(predicted)

def _recall(predicted, relevant, eval_at):
    """
    fraction of the relevant documents that are successfully retrieved
    """
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(set(predicted_at_k).intersection(set(relevant)))
    return n_predicted_and_relevant/ len(relevant)

Nt = 10_000 
Nq = 1
D = 128 
top_k = 100
n_cells = 8
n_subvectors = 64

# 2,000 128-dim vectors for training
np.random.seed(123)
Xtr, Xte = train_test_split(make_blobs(n_samples = Nt, n_features = D)[0].astype(np.float32), test_size=20)
#Xt = np.random.random((Nt, D)).astype(np.float32)  

# the column schema: (name:str, dtype:type, create_index: bool)
pq = PQLite(d_vector=D,
            n_cells=n_cells,
            n_subvectors=n_subvectors,
            columns=[('x', float, True)])

pq.fit(Xtr)
pq.add(Xtr, ids=list(range(len(Xtr))))


query = Xte[[10]]
true_distances = cdist(query, Xtr, metric='euclidean').flatten()
true_ids = np.argsort(true_distances)[0:top_k]
true_dists = true_distances[true_ids]

pq_dists, pq_ids = pq.search(query,  k=top_k)
pq_ids = np.array([int(x) for x in pq_ids[0]])
print('precision: ', _precision(true_ids, pq_ids, top_k))
print('recall: ', _recall(true_ids, pq_ids, top_k))
