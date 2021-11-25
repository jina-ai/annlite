from jina.enums import BetterEnum


class Metric(BetterEnum):
    EUCLIDEAN = 1
    INNER_PRODUCT = 2
    COSINE = 3


class ExpandMode(BetterEnum):
    STEP = 1
    DOUBLE = 2
    ADAPTIVE = 3
