import pytest
from docarray import Document, DocumentArray

from annlite.storage.table import CellTable, MetaTable


@pytest.fixture
def dummy_cell_table():
    table = CellTable(name='dummy', in_memory=True, lazy_create=True)
    table.add_column('name', 'TEXT', create_index=True)
    table.add_column('price', 'FLOAT', create_index=True)
    table.add_column('category', 'TEXT')
    table.create_table()

    return table


@pytest.fixture
def sample_docs():
    return DocumentArray(
        [
            Document(
                id='0', tags={'name': 'orange', 'price': 1.2, 'category': 'fruit'}
            ),
            Document(id='1', tags={'name': 'banana', 'price': 2, 'category': 'fruit'}),
            Document(id='2', tags={'name': 'poly', 'price': 5.1, 'category': 'animal'}),
            Document(id='3', tags={'name': 'bread'}),
        ]
    )


@pytest.fixture
def table_with_data(dummy_cell_table, sample_docs):
    dummy_cell_table.insert(sample_docs)
    return dummy_cell_table


def test_create_cell_table():
    table = CellTable(name='cell_table_x', lazy_create=True)
    table.add_column('x', 'float')
    table.create_table()
    assert table.existed()


def test_schema(dummy_cell_table):
    schema = dummy_cell_table.schema
    assert len(schema.split('\n')) == 5


def test_query(table_with_data):
    result = table_with_data.query(
        where_clause='(category = ?) AND (price < ?)', where_params=('fruit', 3)
    )

    assert len(result) == 2
    assert result[0] == 0


def test_get_docid_by_offset(table_with_data):
    doc_id = table_with_data.get_docid_by_offset(0)
    assert doc_id == '0'

    doc_id = table_with_data.get_docid_by_offset(4)
    assert doc_id is None


def test_exist(table_with_data):
    assert table_with_data.exist('3')


def test_delete(table_with_data):
    table_with_data.delete(['3'])

    assert not table_with_data.exist('3')

    table_with_data.delete_by_offset(2)
    assert not table_with_data.exist('2')


def test_count(table_with_data):
    count = table_with_data.count(
        where_clause='(category = ?) AND (price > ?)', where_params=('fruit', 5)
    )
    assert count == 0

    count = table_with_data.count(
        where_clause='(category = ?) AND (price > ?) and (price < ?)',
        where_params=('fruit', 1, 1.5),
    )
    assert count == 1

    count = table_with_data.count(
        where_clause='(category = ?) AND (price < ?)', where_params=('fruit', 3)
    )
    assert count == 2

    count = table_with_data.count(
        where_clause='(category IN (?, ?)) AND (price < ?)',
        where_params=('fruit', 'animal', 3),
    )
    assert count == 2

    count = table_with_data.count(
        where_clause='(category IN (?)) AND (price < ?)', where_params=('fruit', 1)
    )
    assert count == 0


def test_create_meta_table(tmpdir):
    import datetime

    table = MetaTable('meta_test', data_path=tmpdir)
    addr = table.get_latest_commit()
    assert addr is None

    table.add_address('0', 0, 1)
    table.add_address('2', 1, 5)

    time_since = datetime.datetime.utcnow()
    table.add_address('0', 1, 2)

    assert table.get_address('0') == (1, 2)
    assert table.get_address('2') == (1, 5)

    assert len(list(table.iter_addresses())) == 2
    addresses = list(table.iter_addresses(time_since=time_since))
    assert addresses == [('0', 1, 2)]

    addr = table.get_latest_commit()
    assert addr[:3] == ('0', 1, 2)
    assert addr[-1] >= time_since

    time_since = datetime.datetime.utcnow()
    table.delete_address('0')
    addresses = list(table.iter_addresses(time_since=time_since))
    assert addresses == []
