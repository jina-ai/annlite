from enum import IntEnum


class BetterEnum(IntEnum):
    """The base class of Enum."""

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s: str):
        """
        Parse the enum from a string.
        :param s: string representation of the enum value
        :return: enum value
        """
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(
                f'{s.upper()} is not a valid enum for {cls!r}, must be one of {list(cls)}'
            )


class Metric(BetterEnum):
    EUCLIDEAN = 1
    INNER_PRODUCT = 2
    COSINE = 3


class ExpandMode(BetterEnum):
    STEP = 1
    DOUBLE = 2
    ADAPTIVE = 3
