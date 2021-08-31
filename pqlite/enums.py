from enum import Enum


class Metric(Enum):
    EUCLIDEAN = 1
    INNER_PRODUCT = 2
    COSINE = 3


class ExpandMode(Enum):
    STEP = 1
    DOUBLE = 2
    ADAPTIVE = 3


Metrics = {
    'euclidean': Metric.EUCLIDEAN,
    'inner_product': Metric.INNER_PRODUCT,
    'cosine': Metric.COSINE,
}
