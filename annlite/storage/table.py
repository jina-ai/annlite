import datetime
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from docarray import DocumentArray

sqlite3.register_adapter(np.int64, lambda x: int(x))
sqlite3.register_adapter(np.int32, lambda x: int(x))

COLUMN_TYPE_MAPPING = {
    float: 'FLOAT',
    int: 'INTEGER',
    bool: 'INTEGER',
    str: 'TEXT',
    bytes.__class__: 'BLOB',
    bytes: 'BLOB',
    memoryview: 'BLOB',
    # datetime.datetime: 'TEXT',
    # datetime.date: 'TEXT',
    # datetime.time: 'TEXT',
    None.__class__: 'TEXT',
    # SQLite explicit types
    'TEXT': 'TEXT',
    'INTEGER': 'INTEGER',
    'FLOAT': 'FLOAT',
    'BLOB': 'BLOB',
    'text': 'TEXT',
    'integer': 'INTEGER',
    'float': 'FLOAT',
    'blob': 'BLOB',
}

# If numpy is available, add more types
if np:
    COLUMN_TYPE_MAPPING.update(
        {
            np.int8: 'INTEGER',
            np.int16: 'INTEGER',
            np.int32: 'INTEGER',
            np.int64: 'INTEGER',
            np.uint8: 'INTEGER',
            np.uint16: 'INTEGER',
            np.uint32: 'INTEGER',
            np.uint64: 'INTEGER',
            np.float16: 'FLOAT',
            np.float32: 'FLOAT',
            np.float64: 'FLOAT',
        }
    )


def _converting(value: Any) -> str:
    if isinstance(value, bool):
        if value:
            return 1
        else:
            return 0

    return str(value)


def time_now():
    return datetime.datetime.utcnow()


def _get_table_names(
    conn: 'sqlite3.Connection', fts4: bool = False, fts5: bool = False
) -> List[str]:
    """A list of string table names in this database."""
    where = ["type = 'table'"]
    if fts4:
        where.append("sql like '%USING FTS4%'")
    if fts5:
        where.append("sql like '%USING FTS5%'")
    sql = 'select name from sqlite_master where {}'.format(' AND '.join(where))
    return [r[0] for r in conn.execute(sql).fetchall()]


class Table:
    def __init__(
        self,
        name: str,
        data_path: Optional[Union[Path, str]] = None,
        detect_types: int = 0,
        in_memory: bool = True,
    ):
        if in_memory:
            self._conn_name = ':memory:'
        else:
            if isinstance(data_path, str):
                data_path = Path(data_path)
            self._conn_name = data_path / f'{name}.db'
        self._name = name

        self.detect_types = detect_types

        self._conn = sqlite3.connect(
            self._conn_name, detect_types=detect_types, check_same_thread=False
        )
        self._conn_lock = threading.Lock()

    def execute(self, sql: str, commit: bool = True):
        self._conn.execute(sql)
        if commit:
            self.commit()

    def execute_many(self, sql: str, parameters: List[Tuple], commit: bool = True):
        self._conn.executemany(sql, parameters)
        if commit:
            self.commit()

    def commit(self):
        self._conn.commit()

    def create_table(self):
        ...

    def drop_table(self):
        self._conn.execute(f'DROP table {self.name}')
        self._conn.commit()

    def clear(self):
        """Drop the table and create a new one"""
        self.drop_table()
        self.create_table()

    def load(self, data_file: Union[str, Path]):
        disk_db = sqlite3.connect(data_file, detect_types=self.detect_types)
        disk_db.backup(self._conn)
        disk_db.close()

    def dump(self, data_file: Union[str, Path]):
        backup_db = sqlite3.connect(data_file, detect_types=self.detect_types)
        self._conn.backup(backup_db)
        backup_db.close()

    def close(self):
        self._conn.close()

    @property
    def name(self):
        return self._name

    @property
    def schema(self):
        """SQL schema for this database"""
        result = []
        for row in self._conn.execute(
            f'''PRAGMA table_info("{self.name}")'''
        ).fetchall():
            result.append(', '.join([str(_) for _ in row]))
        return '\n'.join(result)


class CellTable(Table):
    def __init__(
        self,
        name: str,
        columns: Optional[List[tuple]] = None,
        in_memory: bool = True,
        data_path: Optional[Path] = None,
        lazy_create: bool = False,
    ):
        super().__init__(name, data_path=data_path, in_memory=in_memory)

        self._columns = []
        self._indexed_keys = set()

        if columns is not None:
            for name, dtype in columns:
                self.add_column(name, dtype, True)
        if not lazy_create:
            self.create_table()

    @property
    def columns(self) -> List[str]:
        return ['_id', '_doc_id'] + [c.split()[0] for c in self._columns]

    def existed(self):
        return self.name in _get_table_names(self._conn)

    def add_column(self, name: str, dtype: str, create_index: bool = True):
        self._columns.append(f'{name} {COLUMN_TYPE_MAPPING[dtype]}')
        if create_index:
            self._indexed_keys.add(name)

    def create_index(self, column: str, commit: bool = True):
        sql_statement = f'''CREATE INDEX idx_{column}_
                            ON {self.name}(_deleted, {column})'''
        self._conn.execute(sql_statement)

        if commit:
            self._conn.commit()

    def create_table(self):
        sql = f'''CREATE TABLE {self.name}
                    (_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     _doc_id TEXT NOT NULL UNIQUE,
                     _deleted NUMERIC DEFAULT 0'''
        if len(self._columns) > 0:
            sql += ', ' + ', '.join(self._columns)
        sql += ')'
        self._conn.execute(sql)

        sql_statement = f'''CREATE INDEX idx__delete_
                                ON {self.name}(_deleted)'''
        self._conn.execute(sql_statement)

        for name in self._indexed_keys:
            self.create_index(name, commit=False)
        self._conn.commit()

    def insert(
        self,
        docs: 'DocumentArray',
        commit: bool = True,
    ) -> List[int]:
        """Add a single record into the table.

        :param docs: The list of dict docs
        :param commit: If set, commit is applied
        """
        sql_template = 'INSERT INTO {table}({columns}) VALUES ({placeholders});'

        column_names = self.columns[1:]
        columns = ', '.join(column_names)
        placeholders = ', '.join('?' for c in column_names)
        sql = sql_template.format(
            table=self.name, columns=columns, placeholders=placeholders
        )

        values = []
        docs_size = 0
        for doc in docs:
            doc_value = tuple(
                [doc.id]
                + [
                    _converting(doc.tags[c]) if c in doc.tags else None
                    for c in self.columns[2:]
                ]
            )
            values.append(doc_value)
            docs_size += 1

        with self._conn_lock:
            cursor = self._conn.cursor()
            if docs_size > 1:
                cursor.executemany(sql, values[:-1])

            cursor.execute(sql, values[-1])
            last_row_id = cursor.lastrowid
            row_ids = list(range(last_row_id - len(docs), last_row_id))

            if commit:
                self._conn.commit()

        return row_ids

    def query(
        self,
        where_clause: str = '',
        where_params: Tuple = (),
        limit: int = -1,
        offset: int = 0,
        order_by: Optional[str] = None,
        ascending: bool = True,
    ) -> List[int]:
        """Query the records which matches the given conditions

        :param where_clause: where clause for query
        :param where_params: where parameters for query
        :param limit: limit the number of results
        :param offset: offset the number of results
        :param order_by: order the results by the given column
        :param ascending: order the results in ascending or descending order
        :return: offsets list of matched docs
        """

        where_conds = ['_deleted = ?']
        if where_clause:
            where_conds.append(where_clause)
        where = ' and '.join(where_conds)

        _order_by = f'{order_by or "_id"} {"ASC" if ascending else "DESC"}'
        _limit = f'LIMIT {limit}' if limit > 0 else ''
        _offset = f'OFFSET {offset}' if offset > 0 else ''

        sql = f'SELECT _id from {self.name} WHERE {where} ORDER BY {_order_by} {_limit} {_offset}'

        params = (0,) + tuple([_converting(p) for p in where_params])

        # # EXPLAIN SQL query
        # for row in self._conn.execute('EXPLAIN QUERY PLAN ' + sql, params):
        #     print(row)

        # Use `row_factor`
        # https://docs.python.org/3.6/library/sqlite3.html#sqlite3.Connection.row_factory
        def _offset_factory(_, record):
            return record[0] - 1

        self._conn.row_factory = _offset_factory

        cursor = self._conn.cursor()

        try:
            offsets = cursor.execute(sql, params).fetchall()
            self._conn.row_factory = None
            return offsets if offsets else []
        except Exception as e:
            self._conn.row_factory = None
            raise e

    def delete(self, doc_ids: List[str]):
        """Delete the docs

        :param doc_ids: The IDs of docs
        """
        sql = f'UPDATE {self.name} SET _deleted = 1 WHERE _doc_id = ?'
        self._conn.executemany(sql, doc_ids)
        self._conn.commit()

    def get_docid_by_offset(self, offset: int):
        sql = f'SELECT _doc_id from {self.name} WHERE _id = ? and _deleted = 0 LIMIT 1;'
        result = self._conn.execute(sql, (offset + 1,)).fetchone()
        if result:
            return result[0]
        return None

    def delete_by_offset(self, offset: int):
        """Delete the doc with specific offset

        :param offset: The offset of the doc
        """
        sql = f'UPDATE {self.name} SET _deleted = 1 WHERE _id = ?'
        self._conn.execute(sql, (offset + 1,))
        self._conn.commit()

    def undo_delete_by_offset(self, offset: int):
        sql = f'UPDATE {self.name} SET _deleted = 0 WHERE _id = ?'
        self._conn.execute(sql, (offset + 1,))
        self._conn.commit()

    def exist(self, doc_id: str):
        sql = f'SELECT count(*) from {self.name} WHERE _deleted = 0 and _doc_id = ?;'
        return self._conn.execute(sql, (doc_id,)).fetchone()[0] > 0

    def count(self, where_clause: str = '', where_params: Tuple = ()):
        """Return the total number of records which match with the given conditions.
        :param where_clause: where clause for query
        :param where_params: where parameters for query
        :return: the total number of matched records
        """

        if where_clause:
            sql = 'SELECT count(_id) from {table} WHERE {where} LIMIT 1;'
            where = f'_deleted = ? and {where_clause}'
            sql = sql.format(table=self.name, where=where)

            params = (0,) + tuple([_converting(p) for p in where_params])

            # # EXPLAIN SQL query
            # for row in self._conn.execute('EXPLAIN QUERY PLAN ' + sql, params):
            #     print(row)
            return self._conn.execute(sql, params).fetchone()[0]
        else:
            sql = f'SELECT MAX(_id) from {self.name} LIMIT 1;'
            result = self._conn.execute(sql).fetchone()
            if result[0]:
                return result[0] - self.deleted_count()
            return 0

    def deleted_count(self):
        """Return the total number of record what is marked as soft-deleted."""
        sql = f'SELECT count(_id) from {self.name} WHERE _deleted = 1 LIMIT 1'
        return self._conn.execute(sql).fetchone()[0]

    @property
    def size(self):
        return self.count()


class MetaTable(Table):
    def __init__(
        self,
        name: str = 'meta',
        data_path: Optional[Path] = None,
        in_memory: bool = False,
    ):
        super().__init__(
            name,
            data_path=data_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            in_memory=in_memory,
        )
        self.create_table()

    def create_table(self):
        sql = f'''CREATE TABLE if not exists {self.name}
                        (_doc_id TEXT NOT NULL PRIMARY KEY,
                         cell_id INTEGER NOT NULL,
                         offset INTEGER NOT NULL,
                         time_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                         _deleted NUMERIC DEFAULT 0)'''

        self._conn.execute(sql)

        self._conn.execute(
            f'CREATE INDEX if not exists idx_time_at_ ON {self.name}(time_at)'
        )
        self._conn.execute(
            f'CREATE INDEX if not exists idx__delete_ ON {self.name}(_deleted)'
        )

        self._conn.commit()

    def iter_addresses(
        self, time_since: 'datetime.datetime' = datetime.datetime(2020, 2, 2, 0, 0)
    ):
        sql = f'SELECT _doc_id, cell_id, offset from {self.name} WHERE time_at >= ? AND _deleted = 0 ORDER BY time_at ASC;'

        cursor = self._conn.cursor()
        for doc_id, cell_id, offset in cursor.execute(sql, (time_since,)):
            yield doc_id, cell_id, offset

    def get_latest_commit(self):
        sql = f'SELECT _doc_id, cell_id, offset, time_at from {self.name} ORDER BY time_at DESC LIMIT 1;'

        cursor = self._conn.execute(sql)
        row = cursor.fetchone()
        return row

    def get_address(self, doc_id: str):
        sql = f'SELECT cell_id, offset from {self.name} WHERE _doc_id = ? AND _deleted = 0 LIMIT 1;'
        cursor = self._conn.execute(sql, (doc_id,))
        row = cursor.fetchone()
        return (row[0], row[1]) if row else (None, None)

    def delete_address(self, doc_id: str, commit: bool = True):
        sql = f'UPDATE {self.name} SET _deleted = 1, time_at = ? WHERE _doc_id = ?'
        self._conn.execute(
            sql,
            (
                time_now(),
                doc_id,
            ),
        )
        print(f'Deleted {doc_id} at: {time_now()}')
        if commit:
            self._conn.commit()

    def add_address(self, doc_id: str, cell_id: int, offset: int, commit: bool = True):
        sql = f'INSERT OR REPLACE INTO {self.name}(_doc_id, cell_id, offset, time_at, _deleted) VALUES (?, ?, ?, ?, ?);'
        self._conn.execute(
            sql,
            (doc_id, cell_id, offset, time_now(), 0),
        )
        if commit:
            self._conn.commit()

    def bulk_add_address(
        self,
        doc_ids: List[str],
        cell_ids: Union[List[int], np.ndarray],
        offsets: Union[List[int], np.ndarray],
        commit: bool = True,
    ):
        sql = f'INSERT OR REPLACE INTO {self.name}(_doc_id, cell_id, offset, time_at, _deleted) VALUES (?, ?, ?, ?, ?);'
        self._conn.executemany(
            sql,
            [
                (doc_id, cell_id, offset, time_now(), 0)
                for doc_id, cell_id, offset in zip(doc_ids, cell_ids, offsets)
            ],
        )
        if commit:
            self._conn.commit()
