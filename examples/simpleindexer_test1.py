
## Benchmark PQLITE vs SimpleIndexer

import time
import os
import shutil

from jina import Flow
from jina import Document, DocumentArray
from jina.math.distance import cdist
from jina.math.helper import top_k as _top_k
from jina.logging.profile import TimeContext

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

## We can download SimpleIndexer and load it  locally or we can use JinaHub
# from executor_simpleindexer import SimpleIndexer

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
    return n_predicted_and_relevant / len(relevant)

def evaluate(predicts, relevants, eval_at):
    recall = 0
    precision = 0
    for _predict, _relevant in zip(predicts, relevants):
        _predict = np.array([int(x) for x in _predict])
        recall += _recall(_predict, _relevant, top_k)
        precision += _precision(_predict, _relevant, top_k)
    return recall / len(predicts), precision / len(predicts)

def clean_workspace():
    if os.path.exists('./SimpleIndexer'):
        shutil.rmtree('./SimpleIndexer')

    if os.path.exists('./workspace'):
        shutil.rmtree('./workspace')

Nq = 1
D = 128
top_k = 10
n_cells = 64
n_subvectors = 64
n_queries = 1
n_datasets = [1001, 10_001, 500_001, 1_000_001]
times = []

for n_examples in n_datasets:
    clean_workspace()
    np.random.seed(123)
    Xtr, Xte = train_test_split(
        make_blobs(n_samples=n_examples, n_features=D)[0].astype(np.float32), test_size=1
    )

    f = Flow().add(
        uses='jinahub://SimpleIndexer',
        uses_with={'match_args': {'metric': 'euclidean',
                                  'limit': 10}}
    )
    docs = [Document(id='i', embedding=Xtr[i]) for i in range(len(Xtr))]

    with f:
        resp = f.post(
            on='/index',
            inputs=docs,
        )

    with TimeContext(f'\n\nn_examples={n_examples}\n\n'):
        t0 = time.time()
        with f:
            resp = f.post(
                on='/search',
                inputs=DocumentArray([Document(embedding=Xte[0])]),
                return_results=True,
            )
        time_taken = abs(t0 - time.time())
        print(f'time_taken={time_taken}')

clean_workspace()