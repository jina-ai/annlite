import lmdb
import numpy as np


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


def open_lmdb(db_path: str):
    return lmdb.Environment(
        db_path,
        map_size=int(3.436e10),  # in bytes, 32G,
        subdir=False,
        readonly=False,
        metasync=True,
        sync=True,
        map_async=False,
        mode=493,
        create=True,
        readahead=True,
        writemap=False,
        meminit=True,
        max_readers=126,
        max_dbs=0,  # means only one db
        max_spare_txns=1,
        lock=True,
    )
