
import numpy as np
from jina import DocumentArray, Document
from jina.logging.profile import TimeContext
from pqlite import PQLite
import os
import shutil

n_index = [10_000, 100_000, 500_000, 1_000_000]

n_query = [1, 8, 64]
D = 768
R = 5
B = 5000
n_cells = 1
probs =[[0.10, 0.90/3, 0.90/3, 0.90/3],
        [0.30, 0.70/3, 0.70/3, 0.70/3],
        [0.50, 0.50/3, 0.50/3, 0.50/3],
        [0.80, 0.20/3, 0.20/3, 0.20/3]]

times = {}

def clean_workspace():
    if os.path.exists('./data'):
        shutil.rmtree('./data')

    if os.path.exists('./workspace'):
        shutil.rmtree('./workspace')


def docs_with_tags(N, D, probs):
    categories = ['comic', 'movie', 'audiobook', 'shoes']
    X = np.random.random((N, D)).astype(np.float32)
    np.random.seed(123)
    docs = [
        Document(
            id=f'{i}',
            embedding=X[i],
            tags={
                'category': np.random.choice(categories, p=probs),
            },
        )
        for i in range(N)
    ]
    da = DocumentArray(docs)

    return da


results = {}

for n_i in n_index:

    results[n_i] = {}
    for current_probs in probs:
        results[n_i][current_probs[0]] = {}

        times = {}
        clean_workspace()
        columns = [('category', str)]
        idxer = PQLite(
            dim=D,
            initial_size=n_i,
            n_cells=n_cells,
            metas={'workspace': './workspace'},
            columns=columns
        )
        f = {'category': {'$eq': 'comic'}}

        da = docs_with_tags(n_i, D, current_probs)

        with TimeContext(f'indexing {n_i} docs') as t_i:
            for i, _batch in enumerate(da.batch(batch_size=B)):
                idxer.index(_batch)

        times[current_probs[0]] = {}
        times[current_probs[0]]['index'] = t_i.duration

        for n_q in n_query:
            qa = DocumentArray.empty(n_q)
            q_embs = np.random.random([n_q, D]).astype(np.float32)
            qa.embeddings = q_embs
            t_qs = []

            for _ in range(R):
                with TimeContext(f'searching {n_q} docs') as t_q:
                    idxer.search(qa, filter=f)
                t_qs.append(t_q.duration)
            times[current_probs[0]][f'query_{n_q}'] = np.mean(t_qs[1:])  # remove warm-up

        results[n_i][current_probs[0]] = times


title = '| Stored data |% same filter| Indexing time | Query size=1 | Query size=8 | Query size=64|'
print(title)
print('|-----' * 6 + '|')

for n_i in n_index:
    times = results[n_i]

    for current_probs in probs:
        prob = current_probs[0]
        for k, v in times[prob].items():
            s = ' | '.join(f'{v[vv]:.3f}' for vv in ['index', 'query_1', 'query_8', 'query_64'])
            print(f'| {n_i} | {k} | {s} |')
