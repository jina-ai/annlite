from typing import Dict, Any, Iterable, Iterator, List
import sqlite3
import datetime
from jina import Document, DocumentArray
import numpy as np
from ..helper import dumps_doc

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
    datetime.datetime: 'TEXT',
    datetime.date: 'TEXT',
    datetime.time: 'TEXT',
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
    def __init__(self, name: str, in_memory: bool = True):
        if in_memory:
            self._conn_name = ':memory:'
        else:
            self._con_name = f'{name}.db'
        self._name = name

        self._conn = sqlite3.connect(self._conn_name)

    def commit(self):
        self._conn.commit()

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
    def __init__(self, name: str, in_memory: bool = True):
        super(CellTable, self).__init__(name, in_memory=in_memory)

        self._columns = []
        self._indexed_keys = set()

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
                            ON {self.name} ({column})'''
        self._conn.execute(sql_statement)

        if commit:
            self._conn.commit()

    def create_table(self):
        sql = f'''CREATE TABLE {self.name}
                    (_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     _doc_id TEXT NOT NULL UNIQUE,
                     _doc_content BLOB,
                     _deleted NUMERIC DEFAULT 0'''
        if len(self._columns) > 0:
            sql += ', ' + ', '.join(self._columns)
        sql += ')'

        self._conn.execute(sql)
        for name in self._indexed_keys:
            self.create_index(name, commit=False)
        self._conn.commit()

    def insert(
        self,
        docs: DocumentArray,
        commit: bool = True,
    ) -> List[int]:
        """Add a single record into the table.

        :param docs: The list of dict docs
        :param commit: If set, commit is applied
        """
        sql_template = 'INSERT INTO {table}({columns}) VALUES ({placeholders});'
        row_ids = []
        cursor = self._conn.cursor()
        for doc in docs:
            tag_names = [c for c in doc.tags if c in self.columns]
            column_names = ['_doc_id', '_doc_content'] + tag_names
            columns = ', '.join(column_names)
            placeholders = ', '.join('?' for c in column_names)
            sql = sql_template.format(
                table=self.name,
                columns=columns,
                placeholders=placeholders,
            )
            values = tuple([doc.id, dumps_doc(doc)] + [_converting(doc.tags[c]) for c in tag_names])
            cursor.execute(sql, values)
            row_id = cursor.lastrowid
            row_ids.append(row_id)
        if commit:
            self._conn.commit()
        return row_ids

    def query(self, conditions: List[tuple] = []) -> Iterator[dict]:
        """Query the records which matches the given conditions

        :param conditions: the conditions in the format of tuple `(name: str, op: str, value: any)`
        :return: iterator to yield matched doc
        """
        sql = 'SELECT {columns}, _doc_content from {table} WHERE {where} ORDER BY _id ASC;'
        columns = ', '.join(self.columns)

        where_conds = ['_deleted = ?']
        for cond in conditions:
            cond = f'{cond[0]} {cond[1]} ?'
            where_conds.append(cond)
        where = 'and '.join(where_conds)
        sql = sql.format(columns=columns, table=self.name, where=where)

        params = tuple([0] + [_converting(cond[2]) for cond in conditions])

        cursor = self._conn.execute(sql, params)
        keys = [d[0] for d in cursor.description]
        for row in cursor:
            data = dict(zip(keys, row))
            doc = Document(bytes(data['_doc_content']))
            yield data['_id'], doc

    def delete(self, doc_ids: List[str]):
        """Delete the docs

        :param doc_ids: The IDs of docs
        """
        sql = f'UPDATE {self.name} SET _deleted = 1 WHERE _doc_id = ?'
        self._conn.executemany(sql, doc_ids)
        self._conn.commit()

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

    def count(self, conditions: List[tuple] = []):
        """Return the total number of records which match with the given conditions.

        :param conditions: the conditions in the format of tuple `(name: str, op: str, value: any)`
        :return: the total number of matched records
        """
        sql = 'SELECT count(*) from {table} WHERE {where};'
        where_conds = ['_deleted = ?']
        for cond in conditions:
            cond = f'{cond[0]} {cond[1]} ?'
            where_conds.append(cond)
        where = 'and '.join(where_conds)
        sql = sql.format(table=self.name, where=where)
        params = tuple([0] + [_converting(cond[2]) for cond in conditions])
        return self._conn.execute(sql, params).fetchone()[0]


class MetaTable(Table):
    def __init__(self, name: str = 'meta', in_memory: bool = True):
        super(MetaTable, self).__init__(name, in_memory=in_memory)

        sql = f'''CREATE TABLE {self.name}
                (_doc_id TEXT NOT NULL PRIMARY KEY,
                 cell_id INTEGER NOT NULL,
                 offset INTEGER NOT NULL)'''

        self._conn.execute(sql)
        self._conn.commit()

    def get_address(self, doc_id: str):
        sql = f'SELECT cell_id, offset from {self.name} WHERE _doc_id = ?;'
        cursor = self._conn.execute(sql, (doc_id,))
        row = cursor.fetchone()
        return (row[0], row[1]) if row else (None, None)

    def add_address(self, doc_id: str, cell_id: int, offset: int, commit: bool = True):
        sql = f'INSERT OR REPLACE INTO {self.name}(_doc_id, cell_id, offset) VALUES (?, ?, ?);'
        self._conn.execute(
            sql,
            (
                doc_id,
                cell_id,
                offset,
            ),
        )
        if commit:
            self._conn.commit()
