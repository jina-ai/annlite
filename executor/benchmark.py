import numpy as np
from jina import DocumentArray
from jina.logging.profile import TimeContext

n_index = [10_000, 100_000, 500_000, 1_000_000]
n_query = [1, 8, 64]
D = 768
R = 5
B = 4096

from executor import PQLiteIndexer

times = {}

for n_i in n_index:
    idxer = PQLiteIndexer(dim=D, uses_metas={'workspace': './workspace'})
    # build index docs
    i_embs = np.random.random([n_i, D]).astype(np.float32)
    da = DocumentArray.empty(n_i)
    da.embeddings = i_embs

    with TimeContext(f'indexing {n_i} docs') as t_i:
        for _batch in da.batch(batch_size=B):
            idxer.index(_batch)

    times[n_i] = {}
    times[n_i]['index'] = t_i.duration

    for n_q in n_query:
        q_embs = np.random.random([n_q, D]).astype(np.float32)
        qa = DocumentArray.empty(n_q)
        qa.embeddings = q_embs

        t_qs = []
        for _ in range(R):
            with TimeContext(f'searching {n_q} docs') as t_q:
                idxer.search(qa)
            t_qs.append(t_q.duration)
            # # check if it return the full doc
            # assert qa[0].matches
            # assert qa[0].matches.embeddings.shape
        times[n_i][f'query_{n_q}'] = np.mean(t_qs[1:])  # remove warm-up

    idxer.clear()
    idxer.close()

print('|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|')
print('|---' * (len(list(times.values())[0]) + 1) + '|')
for k, v in times.items():
    s = ' | '.join(f'{v[vv]:.3f}' for vv in ['index', 'query_1', 'query_8', 'query_64'])
    print(f'|{k} | {s}|')
