
from jina import DocumentArray, Document
from jina.logging.profile import TimeContext
from pqlite import PQLite

import os
import shutil
import numpy as np

n_index = [10_000, 100_000, 500_000, 1_000_000]

n_query = [1, 8, 64]
D = 768
R = 5
B = 5000
n_cells = 1
probs =[[0.20, 0.30, 0.50],
        [0.05, 0.15, 0.80]]
categories = ['comic', 'movie', 'audiobook']

def clean_workspace():
    if os.path.exists('./data'):
        shutil.rmtree('./data')

    if os.path.exists('./workspace'):
        shutil.rmtree('./workspace')


def docs_with_tags(N, D, probs, categories):

    all_docs = []
    for k,prob in enumerate(probs):
        n_current = int(N*prob)
        X = np.random.random((n_current, D)).astype(np.float32)

        docs = [
            Document(
                embedding=X[i],
                tags={
                    'category': categories[k],
                },
            )
            for i in range(n_current)
        ]
        all_docs.extend(docs)

    return DocumentArray(all_docs)


results = []
for n_i in n_index:

    results_ni = []
    for current_probs in probs:

        clean_workspace()
        columns = [('category', str)]
        idxer = PQLite(
            dim=D,
            initial_size=n_i,
            n_cells=n_cells,
            metas={'workspace': './workspace'},
            columns=columns
        )

        da = docs_with_tags(n_i, D, current_probs, categories)

        with TimeContext(f'indexing {n_i} docs') as t_i:
            for i, _batch in enumerate(da.batch(batch_size=B)):
                idxer.index(_batch)

        for cat,prob in zip(categories, current_probs):
            f = {'category': {'$eq': cat}}

            query_times = []
            for n_q in n_query:
                qa = DocumentArray.empty(n_q)
                q_embs = np.random.random([n_q, D]).astype(np.float32)
                qa.embeddings = q_embs
                t_qs = []

                for _ in range(R):
                    with TimeContext(f'searching {n_q} docs') as t_q:
                        idxer.search(qa, filter=f)
                    t_qs.append(t_q.duration)
                query_times.append(np.mean(t_qs[1:]))

            print(f'\n\nprob={prob}, current_probs={current_probs}, n_i={n_i}\n\n')
            results_ni.append([n_i, prob, t_i.duration] + query_times)

    results.append(results_ni)


title = '| Stored data |% same filter| Indexing time | Query size=1 | Query size=8 | Query size=64|'
print(title)
print('|-----' * 6 + '|')
for block in results:
    sorted_elements_in_block = np.argsort([b[1] for b in block])
    for pos in sorted_elements_in_block:
        res = block[pos]
        print(''.join([f'| {x:.3f} ' for x in res] + ['|']))
