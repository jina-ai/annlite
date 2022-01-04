import numpy as np
from docarray import Document, DocumentArray
from docarray.math.distance import cdist
from docarray.math.helper import top_k as _top_k
from jina.logging.profile import TimeContext

from pqlite import PQLite

from utils import clean_workspace, docs_with_tags, evaluate

n_index = [10_000, 100_000, 500_000, 1_000_000]
n_query = [1, 8, 64]
D = 768
R = 5
B = 100_000
n_cells = 1
probs = [[0.20, 0.30, 0.50],
         [0.05, 0.15, 0.80]]

categories = ['comic', 'movie', 'audiobook']

top_k = 20
n_cells = 1
n_subvectors = D

results = []
for n_i in n_index:

    results_ni = []
    for current_probs in probs:

        clean_workspace()
        columns = [('category', str)]
        indexer = PQLite(
            dim=D,
            initial_size=n_i,
            n_subvectors=n_subvectors,
            n_cells=n_cells,
            metas={'workspace': './workspace'},
            columns=columns
        )

        da = docs_with_tags(n_i, D, current_probs, categories)
        da_embeddings = da.embeddings

        with TimeContext(f'indexing {n_i} docs') as t_i:
            n_train_quantizer = min(n_i,  20_000)
            row_ids = np.random.choice(range(n_i),n_train_quantizer, replace=False)
            indexer.partial_train(da_embeddings[row_ids, :])

            indexer.build_codebook()
            for i, _batch in enumerate(da.batch(batch_size=B)):
                indexer.index(_batch)

        for cat, prob in zip(categories, current_probs):
            f = {'category': {'$eq': cat}}

            indices_cat = np.array([t['category'] for t in da.get_attributes('tags')]) == cat
            ids_indices_cat = np.array([d.id for d in da if d.tags['category'] == cat], dtype='int')

            da_embeddings_cat = da_embeddings[indices_cat, :]

            query_times = []
            for n_q in n_query:
                qa = DocumentArray.empty(n_q)
                q_embs = np.random.random([n_q, D]).astype(np.float32)
                qa.embeddings = q_embs
                t_qs = []

                for _ in range(R):
                    with TimeContext(f'searching {n_q} docs') as t_q:
                        indexer.search(qa, filter=f, limit=top_k)
                    t_qs.append(t_q.duration)

                query_times.append(np.mean(t_qs[1:]))

                if n_q == 8:
                    dists = cdist(q_embs, da_embeddings_cat, metric='euclidean')
                    true_dists, true_local_ids = _top_k(dists, top_k, descending=False)

                    # Note:x
                    #   `true_ids` are not really positions within the original data
                    #   but positions within the subset `da_embeddings_cat`
                    #   We need to go from these positions to the original positions
                    true_ids = ids_indices_cat[true_local_ids]
                    ids = []
                    for doc in qa:
                        ids.append([m.id for m in doc.matches])

                    recall, precision = evaluate(ids, true_ids, top_k)

            print(f'\n\nprob={prob}, current_probs={current_probs}, n_i={n_i}, recall={recall}\n\n')
            results_ni.append([n_i, prob, t_i.duration] + query_times + [recall])

    results.append(results_ni)


title = '| Stored data |% same filter| Indexing time | Query size=1 | Query size=8 | Query size=64| Recall |'
print(title)
print('|-----' * 6 + '|')
for block in results:
    sorted_elements_in_block = np.argsort([b[1] for b in block])
    for pos in sorted_elements_in_block:
        res = block[pos]
        print(''.join([f'| {x:.3f} ' for x in res] + ['|']))
