import tempfile

import numpy as np
import time

from annlite import AnnLite


def get_readable_time(*args, **kwargs):
    """
    Get the datetime in human readable format (e.g. 115 days and 17 hours and 46 minutes and 40 seconds).

    For example:
        .. highlight:: python
        .. code-block:: python
            get_readable_time(seconds=1000)

    :param args: arguments for datetime.timedelta
    :param kwargs: key word arguments for datetime.timedelta
    :return: Datetime in human readable format.
    """
    import math
    import datetime

    secs = float(datetime.timedelta(*args, **kwargs).total_seconds())
    units = [('day', 86400), ('hour', 3600), ('minute', 60), ('second', 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                n = int(secs)
            parts.append(f'{n} {unit}' + ('' if n == 1 else 's'))
    return ' and '.join(parts)


class TimeContext:
    """Timing a code snippet with a context manager."""

    time_attrs = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']

    def __init__(self, task_name: str):
        """
        Create the context manager to timing a code snippet.

        :param task_name: The context/message.
        :param logger: Use existing logger or use naive :func:`print`.

        Example:
        .. highlight:: python
        .. code-block:: python

            with TimeContext('loop'):
                do_busy()

        """
        self.task_name = task_name
        self.duration = 0

    def __enter__(self):
        self.start = time.perf_counter()
        self._enter_msg()
        return self

    def _enter_msg(self):
        print(self.task_name, end=' ...\t', flush=True)

    def __exit__(self, typ, value, traceback):
        self.duration = self.now()

        self.readable_duration = get_readable_time(seconds=self.duration)

        self._exit_msg()

    def now(self) -> float:
        """
        Get the passed time from start to now.

        :return: passed time
        """
        return time.perf_counter() - self.start

    def _exit_msg(self):
        print(
            f'{self.task_name} takes {self.readable_duration} ({self.duration:.2f}s)',
            flush=True,
        )


n_index = [10_000, 100_000, 500_000, 1_000_000]

n_query = [1, 8, 64]
D = 768
R = 5
B = 5000
n_cells = 1
probs = [[0.20, 0.30, 0.50], [0.05, 0.15, 0.80]]
categories = ['comic', 'movie', 'audiobook']


def docs_with_tags(N, D, probs, categories):
    all_docs = []
    for k, prob in enumerate(probs):
        n_current = int(N * prob)
        X = np.random.random((n_current, D)).astype(np.float32)

        docs = [
            dict(
                embedding=X[i],
                category=categories[k],
            )
            for i in range(n_current)
        ]
        all_docs.extend(docs)

    return all_docs


results = []
for n_i in n_index:

    results_ni = []
    for current_probs in probs:
        with tempfile.TemporaryDirectory() as tmpdir:
            columns = [('category', str)]
            idxer = AnnLite(
                D,
                initial_size=n_i,
                n_cells=n_cells,
                columns=columns,
                data_path=tmpdir,
            )

            da = docs_with_tags(n_i, D, current_probs, categories)

            def _batch(l, batch_size):
                for i in range(0, len(l), batch_size):
                    yield l[i : i + batch_size]

            with TimeContext(f'indexing {n_i} docs') as t_i:
                for i, _batch in enumerate(_batch(da, batch_size=B)):
                    idxer.index(_batch)

            for cat, prob in zip(categories, current_probs):
                f = {'category': {'$eq': cat}}

                query_times = []
                for n_q in n_query:
                    q_embs = np.random.random([n_q, D]).astype(np.float32)
                    qa = [dict(embedding=q_embs[i]) for i in range(n_q)]
                    t_qs = []

                    for _ in range(R):
                        with TimeContext(f'searching {n_q} docs') as t_q:
                            idxer.search(qa, filter=f)
                        t_qs.append(t_q.duration)
                    query_times.append(np.mean(t_qs[1:]))

                print(f'\n\nprob={prob}, current_probs={current_probs}, n_i={n_i}\n\n')
                results_ni.append([n_i, int(100 * prob), t_i.duration] + query_times)

    results.append(results_ni)

title = '| Stored data |% same filter| Indexing time | Query size=1  | Query size=8 | Query size=64|'
print(title)
print('|-----' * 6 + '|')
for block in results:
    sorted_elements_in_block = np.argsort([b[1] for b in block])
    for pos in sorted_elements_in_block:
        res = block[pos]
        print(
            ''.join(
                [f'| {x} ' for x in res[0:2]] + [f'| {x:.3f} ' for x in res[2:]] + ['|']
            )
        )
