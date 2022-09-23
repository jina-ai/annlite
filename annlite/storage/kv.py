import shutil
from pathlib import Path
from typing import Dict, List, Union

import lmdb
from docarray import Document, DocumentArray

LMDB_MAP_SIZE = 100 * 1024 * 1024 * 1024


class DocStorage:
    """The backend storage engine of Documents"""

    def __init__(
        self, path: Union[str, Path], serialize_config: Dict = {}, lock: bool = True
    ):
        self._path = path
        self._env = self._open(path, lock=lock)
        self._serialize_config = serialize_config

    def _open(self, db_path: Union[str, Path], lock: bool = True):
        return lmdb.Environment(
            str(self._path),
            map_size=LMDB_MAP_SIZE,
            subdir=True,
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
            lock=lock,
        )

    def insert(self, docs: 'DocumentArray'):
        with self._env.begin(write=True) as txn:
            for doc in docs:
                success = txn.put(
                    doc.id.encode(),
                    doc.to_bytes(**self._serialize_config),
                    overwrite=True,
                )
                if not success:
                    txn.abort()
                    raise ValueError(
                        f'The Doc ({doc.id}) has already been added into database!'
                    )

    def update(self, docs: 'DocumentArray'):
        with self._env.begin(write=True) as txn:
            for doc in docs:
                old_value = txn.replace(
                    doc.id.encode(), doc.to_bytes(**self._serialize_config)
                )
                if not old_value:
                    txn.abort()
                    raise ValueError(f'The Doc ({doc.id}) does not exist in database!')

    def delete(self, doc_ids: List[str]):
        with self._env.begin(write=True) as txn:
            for doc_id in doc_ids:
                txn.delete(doc_id.encode())

    def get(self, doc_ids: Union[str, list]) -> DocumentArray:
        docs = DocumentArray()
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]

        with self._env.begin(write=False) as txn:
            for doc_id in doc_ids:
                buffer = txn.get(doc_id.encode())
                if buffer:
                    doc = Document.from_bytes(buffer, **self._serialize_config)
                    docs.append(doc)
        return docs

    def clear(self):
        self._env.close()
        shutil.rmtree(self._path)
        self._env = self._open(self._path)

    def close(self):
        self._env.close()

    @property
    def stat(self):
        with self._env.begin(write=False) as txn:
            return txn.stat()

    @property
    def size(self):
        return self.stat['entries']

    def batched_iterator(self, batch_size: int = 1, **kwargs):
        with self._env.begin(write=False) as txn:
            count = 0
            docs = DocumentArray()
            cursor = txn.cursor()
            cursor.iternext()
            iterator = cursor.iternext(keys=False, values=True)

            for value in iterator:
                doc = Document.from_bytes(value, **self._serialize_config)
                docs.append(doc)
                count += 1
                if count == batch_size:
                    yield docs
                    count = 0
                    docs = DocumentArray()

            if count > 0:
                yield docs
