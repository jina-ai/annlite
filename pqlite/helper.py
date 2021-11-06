from typing import List
import numpy as np

from jina import Document


def dumps_doc(doc: Document):
    new_doc = Document(doc, copy=True)
    return new_doc.SerializeToString()


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
