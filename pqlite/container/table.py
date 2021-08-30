from typing import Dict, Any, Iterable, List
import sqlite3
import datetime

COLUMN_TYPE_MAPPING = {
    float: "FLOAT",
    int: "INTEGER",
    bool: "INTEGER",
    str: "TEXT",
    bytes.__class__: "BLOB",
    bytes: "BLOB",
    memoryview: "BLOB",
    datetime.datetime: "TEXT",
    datetime.date: "TEXT",
    datetime.time: "TEXT",
    None.__class__: "TEXT",
    # SQLite explicit types
    "TEXT": "TEXT",
    "INTEGER": "INTEGER",
    "FLOAT": "FLOAT",
    "BLOB": "BLOB",
    "text": "TEXT",
    "integer": "INTEGER",
    "float": "FLOAT",
    "blob": "BLOB",
}


def _converting(value: Any) -> str:
    if isinstance(value, bool):
        if value:
            return 1
        else:
            return 0
    return str(value)


class Table:
    def __init__(self, name: str, in_memory: bool = True):
        if in_memory:
            self._conn_name = ':memory:'
        else:
            self._con_name = f'{name}.db'
        self._name = name
        self._conn = sqlite3.connect(self._conn_name)
        self._cursor = self._conn.cursor()

        self._columns = []
        self._index_keys = set()

        self._is_created = False

    @property
    def name(self):
        return self._name

    @property
    def column_names(self):
        return ['_id', '_doc_id'] + [c.split()[0] for c in self._columns]

    @property
    def table_names(self, fts4: bool = False, fts5: bool = False) -> List[str]:
        """A list of string table names in this database."""
        where = ["type = 'table'"]
        if fts4:
            where.append("sql like '%USING FTS4%'")
        if fts5:
            where.append("sql like '%USING FTS5%'")
        sql = "select name from sqlite_master where {}".format(" AND ".join(where))
        return [r[0] for r in self.execute(sql).fetchall()]

    def existed(self):
        return self.name in self.table_names

    def add_column(self, name: str, dtype: str, is_key: bool = True):
        self._columns.append(f'{name} {dtype.upper()}')
        if is_key:
            self._index_keys.add(name)

    def create_index(self, column_name: str, commit: bool = False):
        sql_statement = f'''CREATE INDEX idx_{column_name} 
                            ON {self.name} ({column_name})'''
        self._conn.execute(sql_statement)

        if commit:
            self._conn.commit()

    def create_table(self):
        with self._conn:
            sql_statement = f'''CREATE TABLE {self.name} 
                                    (_id INTEGER NOT NULL PRIMARY KEY, 
                                     _doc_id TEXT NOT NULL, 
                                     _deleted NUMERIC DEFAULT 0,
                                     {', '.join(self._columns)})'''
            self._conn.execute(sql_statement)

            for name in self._index_keys:
                self.create_index(name)

        self._is_created = True

    def insert(
        self,
        docs: Iterable[Dict[str, Any]],
        replace: bool = False,
        ignore: bool = False,
    ):
        """Insert a single record into the table."""
        or_what = ''
        if replace:
            or_what = 'OR REPLACE '
        elif ignore:
            or_what = 'OR IGNORE '

        sql_statement = (
            'INSERT {or_what}INTO {table}({columns}) VALUES ({placeholders});'
        )
        with self._conn:
            for doc in docs:
                column_names = [c for c in doc.keys() if c in self.column_names]
                columns = ', '.join(column_names)
                placeholders = ', '.join('?' for c in column_names)
                sql = sql_statement.format(
                    or_what=or_what,
                    table=self.name,
                    columns=columns,
                    placeholders=placeholders,
                )
                values = tuple([_converting(doc[c]) for c in column_names])
                self._conn.execute(sql, values)

    def query(self, conditions: List):
        sql = 'SELECT {columns} from {table} WHERE {where} ORDER BY _id ASC;'
        columns = ', '.join(self.column_names)

        conds = []
        for cond in conditions:
            cond = f'{cond[0]} {cond[1]} ?'
            conds.append(cond)
        where = 'and '.join(conds)
        sql = sql.format(columns=columns, table=self.name, where=where)

        params = tuple([_converting(cond[2]) for cond in conditions])
        cursor = self._conn.execute(sql, params)
        keys = [d[0] for d in cursor.description]
        for row in cursor:
            doc = dict(zip(keys, row))
            doc['_id'] -= 1
            yield doc

    @property
    def schema(self):
        """SQL schema for this database"""
        result = []
        for row in self._cursor.execute(
            f'''PRAGMA table_info("{self.name}")'''
        ).fetchall():
            result.append(', '.join([str(_) for _ in row]))
        return '\n'.join(result)
