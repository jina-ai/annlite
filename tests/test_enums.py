import pytest

from annlite.enums import Metric


def test_metric():
    m = Metric.EUCLIDEAN

    assert m.name == 'EUCLIDEAN'
    assert m.value == 1
