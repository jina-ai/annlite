import os
import shutil

import numpy as np
from docarray import Document, DocumentArray


def clean_workspace():
    if os.path.exists('./data'):
        shutil.rmtree('./data')

    if os.path.exists('./workspace'):
        shutil.rmtree('./workspace')


def docs_with_tags(N, D, probs, categories):

    all_docs = []
    start_current = 0
    for k, prob in enumerate(probs):
        n_current = int(N * prob)
        X = np.random.random((n_current, D)).astype(np.float32)

        docs = [
            Document(
                embedding=X[i],
                id=f'{i+start_current}',
                tags={
                    'category': categories[k],
                },
            )
            for i in range(n_current)
        ]
        all_docs.extend(docs)
        start_current += n_current

    return DocumentArray(all_docs)


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


def evaluate(predicts, relevants, top_k):
    recall = 0
    precision = 0
    for _predict, _relevant in zip(predicts, relevants):
        _predict = np.array([int(x) for x in _predict])
        recall += _recall(_predict, _relevant, top_k)
        precision += _precision(_predict, _relevant, top_k)

    return recall / len(predicts), precision / len(predicts)
