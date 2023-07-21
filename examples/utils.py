import os
import shutil
import time
import numpy as np


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


def _batch(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i : i + batch_size]


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
            dict(
                embedding=X[i],
                id=f'{i+start_current}',
                category=categories[k],
            )
            for i in range(n_current)
        ]
        all_docs.extend(docs)
        start_current += n_current

    return all_docs


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
