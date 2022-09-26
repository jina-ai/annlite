import sys

import numpy as np
from loguru import logger


def setup_logging(debug: bool):
    """
    Setup the log formatter for AnnLite.
    """

    log_level = 'INFO'
    if debug:
        log_level = 'DEBUG'

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        level=log_level,
    )


def str2dtype(dtype_str: str):
    if dtype_str in ['double', 'float64']:
        dtype = np.float64
    elif dtype_str in ['half', 'float16']:
        dtype = np.float16
    elif dtype_str in ['float', 'float32']:
        dtype = np.float32
    elif dtype_str in ['bfloat16']:
        dtype = np.bfloat16
    elif dtype_str in ['long', 'int64']:
        dtype = np.int64
    elif dtype_str in ['int', 'int32']:
        dtype = np.int32
    elif dtype_str in ['int16']:
        dtype = np.int16
    elif dtype_str in ['int8']:
        dtype = np.int8
    elif dtype_str in ['uint8']:
        dtype = np.uint8
    elif dtype_str in ['bool']:
        dtype = np.bool
    else:
        raise TypeError(f'Unrecognized dtype string: {dtype_str}')
    return dtype
