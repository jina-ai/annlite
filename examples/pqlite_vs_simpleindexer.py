
## Benchmark PQLITE vs SimpleIndexer
import time
import os
import shutil

from jina import Flow
import numpy as np
from jina import Document, DocumentArray
from jina.math.distance import cdist
from jina.math.helper import top_k as _top_k
from executor_pqlite import PQLiteIndexer

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

def create_data(n_examples, D):
    np.random.seed(123)
    Xtr, Xte = train_test_split(
        make_blobs(n_samples=n_examples, n_features=D)[0].astype(np.float32), test_size=1
    )
    return Xtr, Xte

"""
Stored data	Indexing time	Query size=1	Query size=8	Query size=64
10000	     0.256	         0.019	        0.029        	0.086
50000	     1.156	         0.147	        0.177        	0.314
100000	     2.329	         0.297	        0.332        	0.536
200000	     4.704	         0.656	        0.744        	1.050
400000	     11.105          1.289	        1.536        	2.793
"""

Nq = 1
D = 128
top_k = 10
n_cells = 64
n_subvectors = 64
n_queries = 1
n_datasets = [10001, 50001, 200001, 400001 ]

BENCHMARK_SIMPLEINDEXER = False
BENCHMARK_PQLITE = True


if BENCHMARK_SIMPLEINDEXER:

    ################ SimpleIndexer Benchmark BEGIN #################
    times = []

    for n_examples in n_datasets:
        time_taken = 0
        clean_workspace()
        Xtr, Xte = create_data(n_examples, D)

        f = Flow().add(
            uses='jinahub://SimpleIndexer',
            uses_with={'match_args': {'metric': 'euclidean',
                                      'limit': 10}}
        )
        docs = [Document(id=f'{i}', embedding=Xtr[i]) for i in range(len(Xtr))]

        with f:
            resp = f.post(
                on='/index',
                inputs=docs,
            )

        with f:
            t0 = time.time()
            resp = f.post(
                on='/search',
                inputs=DocumentArray([Document(embedding=Xte[0])]),
                return_results=True,
            )
            time_taken = time.time() - t0

        times.append(time_taken)


    df = pd.DataFrame({'n_examples': n_datasets, 'times':times})
    df.to_csv('simpleindexer.csv')
    print(df)
    clean_workspace()
    ################ SimpleIndexer Benchmark END #################


if BENCHMARK_PQLITE:

    ################ PqLite Benchmark BEGIN ######################
    n_datasets = [10001, 50001, 200001, 400001, 1_000_001, 5_000_001]
    times = []

    for n_examples in n_datasets:
        time_taken = 0
        clean_workspace()
        Xtr, Xte = create_data(n_examples, D)

        f = Flow().add(
            uses=PQLiteIndexer,
            uses_with={
                'dim': D,
                'limit':10,
            },
            uses_metas=metas,
        )
        docs = [Document(id=f'{i}', embedding=Xtr[i]) for i in range(len(Xtr))]

        with f:
            resp = f.post(
                on='/index',
                inputs=docs,
            )

        with f:
            t0 = time.time()
            resp = f.post(
                on='/search',
                inputs=DocumentArray([Document(embedding=Xte[0])]),
                return_results=True,
            )
            time_taken = time.time() - t0

        times.append(time_taken)


    df = pd.DataFrame({'n_examples': n_datasets, 'times':times})
    df.to_csv('pqlite.csv')
    print(df)
    clean_workspace()
    ################ PqLite Benchmark END #########################

