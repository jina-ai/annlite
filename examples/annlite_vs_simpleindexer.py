import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
from jina import Document, DocumentArray, Flow
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from executor.executor import AnnLiteIndexer

Nq = 1
D = 128
top_k = 10
R = 5
n_cells = 64
n_subvectors = 64
n_queries = 1

BENCHMARK_SIMPLEINDEXER = False
BENCHMARK_ANNLITE = True


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


def create_data(n_examples, D):
    np.random.seed(123)
    Xtr, Xte = train_test_split(
        make_blobs(n_samples=n_examples, n_features=D)[0].astype(np.float32),
        test_size=1,
    )
    return Xtr, Xte


def create_data_online(n_examples, D, batch_size):
    np.random.seed(123)
    num = 0
    while True:
        Xtr_batch = make_blobs(n_samples=batch_size, n_features=D)[0].astype(np.float32)
        yield DocumentArray([Document(embedding=x) for x in Xtr_batch])
        num += batch_size

        if num + batch_size >= n_examples:
            break

    if num < n_examples:
        Xtr_batch = make_blobs(n_samples=n_examples - num, n_features=D)[0].astype(
            np.float32
        )
        yield DocumentArray([Document(embedding=x) for x in Xtr_batch])


def create_test_data(D, Nq):
    np.random.seed(123)
    Xte = make_blobs(n_samples=Nq, n_features=D)[0].astype(np.float32)
    return DocumentArray([Document(embedding=x) for x in Xte])


if BENCHMARK_SIMPLEINDEXER:

    ################ SimpleIndexer Benchmark BEGIN #################
    n_datasets = [10001, 50001, 200001, 400001]
    times = []

    for n_examples in n_datasets:
        time_taken = 0

        Xtr, Xte = create_data(n_examples, D)

        with tempfile.TemporaryDirectory() as tmpdir:

            f = Flow().add(
                uses='jinahub://SimpleIndexer',
                uses_with={'match_args': {'metric': 'euclidean', 'limit': 10}},
                workspace=tmpdir,
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

    df = pd.DataFrame({'n_examples': n_datasets, 'times': times})
    df.to_csv('simpleindexer.csv')
    print(df)
    ################ SimpleIndexer Benchmark END #################


if BENCHMARK_ANNLITE:

    ################ AnnLite Benchmark BEGIN ######################
    n_datasets = [10_000, 100_000, 500_000, 1_000_000, 10_000_000]
    # n_datasets = [10_000, 100_000]
    n_queries = [1, 8, 64]

    batch_size = 4096
    times = []

    results = {}
    for n_examples in n_datasets:
        print(f'\n\nWorking with n_examples={n_examples}\n\n')
        time_taken = 0

        with tempfile.TemporaryDirectory() as tmpdir:

            f = Flow().add(
                uses=AnnLiteIndexer,
                uses_with={
                    'n_dim': D,
                    'limit': 10,
                },
                workspace=tmpdir,
            )

            docs = create_data_online(n_examples, D, batch_size)

            results_current = {}
            with f:
                time_taken = 0
                for batch in docs:
                    t0 = time.time()
                    resp = f.post(on='/index', inputs=batch, request_size=10240)
                    # This is done to avoid data creation time loaded in index time
                    time_taken += time.time() - t0
                results_current['index_time'] = time_taken

            times_per_n_query = []
            with f:
                for n_query in n_queries:
                    da_queries = create_test_data(D, n_query)
                    t_qs = []
                    for _ in range(R):
                        t0 = time.time()
                        resp = f.post(
                            on='/search',
                            inputs=da_queries,
                            return_results=True,
                        )
                        time_taken = time.time() - t0
                        t_qs.append(time_taken)
                    # remove warm-up
                    times_per_n_query.append(np.mean(t_qs[1:]))

            results_current['query_times'] = times_per_n_query
            print(f'==> query_times: {times_per_n_query}')
            df = pd.DataFrame({'results': results_current})
            df.to_csv(f'annlite_{n_examples}.csv')
            results[n_examples] = results_current

    df = pd.DataFrame(results)
    df.to_csv('annlite.csv')
    clean_workspace()
    ################ AnnLite Benchmark END #########################
