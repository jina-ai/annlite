
import numpy as np
from jina import DocumentArray, Document
from jina.logging.profile import TimeContext
from executor import PQLiteIndexer
from pqlite.filter import Filter

n_index = [10_000, 100_000, 500_000, 1_000_000]
n_index = [5000, 10_000]

n_query = [1, 8, 64]
D = 768
R = 5
B = 4096
n_cells = 1
probs =[[0.05, 0.95/3, 0.95/3, 0.95/3],
        [0.10, 0.90/3, 0.90/3, 0.90/3],
        [0.30, 0.70/3, 0.70/3, 0.70/3],
        [0.50, 0.50/3, 0.50/3, 0.50/4],
        [0.80, 0.20/3, 0.20/3, 0.20/3]]

times = {}

def docs_with_tags(N, D, probs):
    categories = ['comic', 'movie', 'audiobook', 'shoes']
    X = np.random.random((N, D)).astype(np.float32)
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

for n_i in n_index:


    columns = [ ('category', 'str')]
    idxer = PQLiteIndexer(
        dim=D,
        initial_size=n_i,
        n_cells=n_cells,
        metas={'workspace': './workspace'},
        columns=columns
    )
    f = {'category': {'$eq': 'comic'}}

    for current_probs in probs:

        da = docs_with_tags(n_i, D, current_probs)
        with TimeContext(f'indexing {n_i} docs') as t_i:
            for _batch in da.batch(batch_size=B):
                idxer.index(_batch)

        times[n_i] = {}
        times[n_i][current_probs[0]]
        times[n_i][current_probs[0]]['index'] = t_i.duration

        for n_q in n_query:
            q_embs = np.random.random([n_q, D]).astype(np.float32)
            qa = DocumentArray.empty(n_q)
            qa.embeddings = q_embs

            t_qs = []

            for _ in range(R):
                with TimeContext(f'searching {n_q} docs') as t_q:
                    idxer.search(qa, filter=f)
                t_qs.append(t_q.duration)
            times[n_i][current_probs[0]][f'query_{n_q}'] = np.mean(t_qs[1:])  # remove warm-up

    idxer.clear()
    idxer.close()

print('|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|')
print('|---' * (len(list(times.values())[0]) + 1) + '|')
for k, v in times.items():
    s = ' | '.join(f'{v[vv]:.3f}' for vv in ['index', 'query_1', 'query_8', 'query_64'])
    print(f'|{k} | {s}|')
